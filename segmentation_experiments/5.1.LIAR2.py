import pandas as pd
import numpy as np
import os
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, BertTokenizer

# Fixing the randomness of CUDA.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("PyTorch Version : {}".format(torch.__version__))
print(DEVICE)


worksapce = ''
USE_PRETRAINED_JUSTIFICATION = os.environ.get('LIAR2_PRETRAINED_JUSTIFICATION', '0') == '1'
USE_ENGINEERED_NUMERICAL_FEATURES = os.environ.get('LIAR2_ENGINEERED_NUMERICAL_FEATURES', '0') == '1'
USE_STATEMENT_BRANCH = os.environ.get('LIAR2_USE_STATEMENT_BRANCH', '1') == '1'
LABEL_SMOOTHING = float(os.environ.get('LIAR2_LABEL_SMOOTHING', '0'))
ORDINAL_LOSS_WEIGHT = float(os.environ.get('LIAR2_ORDINAL_LOSS_WEIGHT', '0'))
ENSEMBLE_SIZE = max(1, int(os.environ.get('LIAR2_ENSEMBLE_SIZE', '1')))
ENSEMBLE_SEED = 42
JUSTIFICATION_MODEL_NAME = 'distilbert-base-uncased'
JUSTIFICATION_MAX_LENGTH = 192
BASE_TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
JUSTIFICATION_TOKENIZER = (
    AutoTokenizer.from_pretrained(JUSTIFICATION_MODEL_NAME)
    if USE_PRETRAINED_JUSTIFICATION else BASE_TOKENIZER
)
if USE_PRETRAINED_JUSTIFICATION and USE_ENGINEERED_NUMERICAL_FEATURES and not USE_STATEMENT_BRANCH:
    model_save = 'LIAR2_DistilBERT_engineered_no_statement.pt'
elif USE_PRETRAINED_JUSTIFICATION:
    model_save = 'LIAR2_DistilBERT_justification.pt'
else:
    model_save = 'LIAR2.pt'
if ENSEMBLE_SIZE > 1:
    model_save = model_save.replace('.pt', '_ensemble.pt')
model_name = 'LIAR2'
num_epochs = 10
batch_size = 16 if USE_PRETRAINED_JUSTIFICATION else 32
learning_rate = 3e-4 if USE_PRETRAINED_JUSTIFICATION else 1e-3
encoder_learning_rate = 2e-5
early_stopping_patience = 3
num_classes = 6
padding_idx = 0
metadata_each_dim = 10
NUMERICAL_COLUMNS = [
    'true_counts', 'mostly_true_counts', 'half_true_counts',
    'mostly_false_counts', 'false_counts', 'pants_on_fire_counts'
]
NUMERIC_FEATURE_DIM = len(NUMERICAL_COLUMNS) * (2 if USE_ENGINEERED_NUMERICAL_FEATURES else 1) + int(USE_ENGINEERED_NUMERICAL_FEATURES)

print(model_save)

# col = ['id', 'label', 'statement', 'subject', 'speaker', 'job_title', 'state_info', 'party_affiliation', 'barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context']
col = ["id", "label", "statement", "date", "subject", "speaker", "speaker_description", "state_info", "true_counts", "mostly_true_counts", "half_true_counts", "mostly_false_counts", "false_counts", "pants_on_fire_counts", "context", "justification"]

label_map = {0: 'pants-fire', 1: 'false', 2: 'barely-true', 3: 'half-true', 4: 'mostly-true', 5: 'true'}
label_convert = {'pants-fire': 0, 'false': 1, 'barely-true': 2, 'half-true': 3, 'mostly-true': 4, 'true':5}

train_data = pd.read_csv(worksapce + 'train.csv')
test_data = pd.read_csv(worksapce + 'test.csv')
val_data = pd.read_csv(worksapce + 'valid.csv')

# Replace NaN values with 'NaN'
train_data[NUMERICAL_COLUMNS] = train_data[NUMERICAL_COLUMNS].fillna(0)
train_data.fillna('NaN', inplace=True)

test_data[NUMERICAL_COLUMNS] = test_data[NUMERICAL_COLUMNS].fillna(0)
test_data.fillna('NaN', inplace=True)

val_data[NUMERICAL_COLUMNS] = val_data[NUMERICAL_COLUMNS].fillna(0)
val_data.fillna('NaN', inplace=True)


def textProcess(input_text, max_length = -1):
    if max_length == -1:
        tokens = BASE_TOKENIZER(input_text, truncation=True, padding=True)
    else:
        tokens = BASE_TOKENIZER(input_text, truncation=True, padding='max_length', max_length=max_length)
    return tokens


def justificationProcess(input_text):
    if USE_PRETRAINED_JUSTIFICATION:
        return JUSTIFICATION_TOKENIZER(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=JUSTIFICATION_MAX_LENGTH,
            return_tensors='pt'
        )
    return JUSTIFICATION_TOKENIZER(input_text, truncation=True, padding=True)


def prepare_numeric_features(train_df, val_df, test_df):
    def get_counts(df):
        counts = df[NUMERICAL_COLUMNS].apply(pd.to_numeric, errors='coerce').fillna(0)
        return counts.clip(lower=0).astype(np.float32)

    train_counts = get_counts(train_df)
    val_counts = get_counts(val_df)
    test_counts = get_counts(test_df)

    if not USE_ENGINEERED_NUMERICAL_FEATURES:
        return tuple(counts.to_numpy(dtype=np.float32) for counts in (train_counts, val_counts, test_counts))

    def engineer(counts):
        total = counts.sum(axis=1)
        ratios = counts.div(total.where(total > 0, 1), axis=0)
        features = pd.concat(
            [
                np.log1p(counts).add_prefix('log_'),
                ratios.add_prefix('ratio_'),
                np.log1p(total).rename('log_total')
            ],
            axis=1
        )
        return features.astype(np.float32)

    train_features = engineer(train_counts)
    val_features = engineer(val_counts)
    test_features = engineer(test_counts)
    feature_mean = train_features.mean(axis=0)
    feature_std = train_features.std(axis=0).replace(0, 1)
    return tuple(
        ((features - feature_mean) / feature_std).to_numpy(dtype=np.float32)
        for features in (train_features, val_features, test_features)
    )


train_numeric_features, val_numeric_features, test_numeric_features = prepare_numeric_features(
    train_data, val_data, test_data
)


# Define a custom dataset for loading the data
class LiarDataset(data.Dataset):
    def __init__(self, data_df, statement, label_onehot, label, date, subject, speaker, speaker_description, state_info,
                    numeric_features, context, justification, justification_attention_mask):
        self.data_df = data_df
        self.statement = statement
        self.label_onehot = label_onehot
        self.label = label
        self.justification = justification
        self.justification_attention_mask = justification_attention_mask
        self.metadata_text = torch.cat((date.int(), subject.int(), speaker.int(), speaker_description.int(), state_info.int(), context.int()), dim=-1)
        self.metadata_number = torch.as_tensor(numeric_features, dtype=torch.float32)

    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        statement = self.statement[idx]
        label_onehot = self.label_onehot[idx]
        label = self.label[idx]
        justification = self.justification[idx]
        justification_attention_mask = self.justification_attention_mask[idx]
        metadata_text = self.metadata_text[idx]
        metadata_number = self.metadata_number[idx]
        return statement, label_onehot, label, metadata_text, metadata_number, justification, justification_attention_mask


# Define the data loaders for training and validation
train_text = torch.tensor(textProcess(train_data['statement'].tolist())['input_ids'])
train_justification_tokens = justificationProcess(train_data['justification'].tolist())
train_justification = torch.as_tensor(train_justification_tokens['input_ids'])
train_justification_attention_mask = torch.as_tensor(train_justification_tokens['attention_mask'])
train_label = torch.nn.functional.one_hot(torch.tensor(train_data['label'].replace(label_convert)), num_classes=6).float()
train_date = torch.tensor(textProcess(train_data['date'].tolist(), metadata_each_dim)['input_ids'])
train_subject = torch.tensor(textProcess(train_data['subject'].tolist(), metadata_each_dim)['input_ids'])
train_speaker = torch.tensor(textProcess(train_data['speaker'].tolist(), metadata_each_dim)['input_ids'])
train_speaker_description = torch.tensor(textProcess(train_data['speaker_description'].tolist(), metadata_each_dim)['input_ids'])
train_state_info = torch.tensor(textProcess(train_data['state_info'].tolist(), metadata_each_dim)['input_ids'])
train_context = torch.tensor(textProcess(train_data['context'].tolist(), metadata_each_dim)['input_ids'])

train_dataset = LiarDataset(train_data, train_text, train_label, torch.tensor(train_data['label'].replace(label_convert)), 
                            train_date, train_subject, train_speaker, train_speaker_description, train_state_info, 
                            train_numeric_features, train_context,
                            train_justification, train_justification_attention_mask)
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_text = torch.tensor(textProcess(val_data['statement'].tolist())['input_ids'])
val_justification_tokens = justificationProcess(val_data['justification'].tolist())
val_justification = torch.as_tensor(val_justification_tokens['input_ids'])
val_justification_attention_mask = torch.as_tensor(val_justification_tokens['attention_mask'])
val_label = torch.nn.functional.one_hot(torch.tensor(val_data['label'].replace(label_convert)), num_classes=6).float()
val_date = torch.tensor(textProcess(val_data['date'].tolist(), metadata_each_dim)['input_ids'])
val_subject = torch.tensor(textProcess(val_data['subject'].tolist(), metadata_each_dim)['input_ids'])
val_speaker = torch.tensor(textProcess(val_data['speaker'].tolist(), metadata_each_dim)['input_ids'])
val_speaker_description = torch.tensor(textProcess(val_data['speaker_description'].tolist(), metadata_each_dim)['input_ids'])
val_state_info = torch.tensor(textProcess(val_data['state_info'].tolist(), metadata_each_dim)['input_ids'])
val_context = torch.tensor(textProcess(val_data['context'].tolist(), metadata_each_dim)['input_ids'])

val_dataset = LiarDataset(val_data, val_text, val_label, torch.tensor(val_data['label'].replace(label_convert)),
                          val_date, val_subject, val_speaker, val_speaker_description, val_state_info, 
                          val_numeric_features, val_context,
                          val_justification, val_justification_attention_mask)
val_loader = data.DataLoader(val_dataset, batch_size=batch_size)

test_text = torch.tensor(textProcess(test_data['statement'].tolist())['input_ids'])
test_justification_tokens = justificationProcess(test_data['justification'].tolist())
test_justification = torch.as_tensor(test_justification_tokens['input_ids'])
test_justification_attention_mask = torch.as_tensor(test_justification_tokens['attention_mask'])
test_label = torch.nn.functional.one_hot(torch.tensor(test_data['label'].replace(label_convert)), num_classes=6).float()
test_date = torch.tensor(textProcess(test_data['date'].tolist(), metadata_each_dim)['input_ids'])
test_subject = torch.tensor(textProcess(test_data['subject'].tolist(), metadata_each_dim)['input_ids'])
test_speaker = torch.tensor(textProcess(test_data['speaker'].tolist(), metadata_each_dim)['input_ids'])
test_speaker_description = torch.tensor(textProcess(test_data['speaker_description'].tolist(), metadata_each_dim)['input_ids'])
test_state_info = torch.tensor(textProcess(test_data['state_info'].tolist(), metadata_each_dim)['input_ids'])
test_context = torch.tensor(textProcess(test_data['context'].tolist(), metadata_each_dim)['input_ids'])

test_dataset = LiarDataset(test_data, test_text, test_label, torch.tensor(test_data['label'].replace(label_convert)),
                          test_date, test_subject, test_speaker, test_speaker_description, test_state_info, 
                          test_numeric_features, test_context,
                          test_justification, test_justification_attention_mask)
test_loader = data.DataLoader(test_dataset, batch_size=batch_size)


class FuzzyLayer(nn.Module):
    def __init__(self, input_dim, membership_num):
        super(FuzzyLayer, self).__init__()

        # input_dim: feature number of the dataset
        # membership_num: number of membership function, also known as the class number

        self.input_dim = input_dim
        self.membership_num = membership_num

        self.membership_miu = nn.Parameter(torch.Tensor(self.membership_num, self.input_dim).to(DEVICE), requires_grad=True)
        self.membership_sigma = nn.Parameter(torch.Tensor(self.membership_num, self.input_dim).to(DEVICE), requires_grad=True)

        nn.init.xavier_uniform_(self.membership_miu)
        nn.init.ones_(self.membership_sigma)

    def forward(self, input_seq):
        batch_size = input_seq.size()[0]
        input_seq_exp = input_seq.unsqueeze(1).expand(batch_size, self.membership_num, self.input_dim)
        membership_miu_exp = self.membership_miu.unsqueeze(0).expand(batch_size, self.membership_num, self.input_dim)
        membership_sigma_exp = self.membership_sigma.unsqueeze(0).expand(batch_size, self.membership_num, self.input_dim)

        fuzzy_membership = torch.mean(torch.exp((-1 / 2) * ((input_seq_exp - membership_miu_exp) / membership_sigma_exp) ** 2), dim=-1)
        return fuzzy_membership



class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.convs = nn.ModuleList([
                                    nn.Conv1d(in_channels = embedding_dim, 
                                              out_channels = n_filters, 
                                              kernel_size = fs)
                                    for fs in filter_sizes
                                    ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text):
        #text = [batch size, sent len]
        embedded = self.embedding(text)
        #embedded = [batch size, sent len, emb dim]

        embedded = embedded.permute(0, 2, 1)
        #embedded = [batch size, emb dim, sent len]

        conved = [F.relu(conv(embedded)) for conv in self.convs]
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        #pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim = 1))
        #cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)

class CNNBiLSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()

        self.embedding = nn.Linear(input_dim, embedding_dim)
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=32, kernel_size=1)
        self.rnn = nn.LSTM(32, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, metadata):
        #metadata = [batch size, metadata dim]

        embedded = self.dropout(self.embedding(metadata))
        #embedded = [batch size, metadata dim, emb dim]

        embedded = torch.reshape(embedded, (metadata.size(0), 128, 1))

        conved = F.relu(self.conv(embedded))
        #conved = [batch size, n_filters, metadata dim - filter_sizes[n] + 1]

        conved = torch.reshape(conved, (metadata.size(0), 32))

        outputs, (hidden, cell) = self.rnn(conved)
        #outputs = [metadata dim - filter_sizes[n] + 1, batch size, hid dim * num directions]
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]

        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        #and apply dropout
        # hidden = self.dropout(torch.cat((hidden[-1,:], hidden[0,:]), dim = -1))
        #hidden = [batch size, hid dim * num directions]

        return self.fc(outputs)


class DistilBertJustification(nn.Module):
    def __init__(self, output_dim, dropout):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(JUSTIFICATION_MODEL_NAME)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask):
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = encoded.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(pooled))


class LiarModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, padding_idx, input_dim, input_dim_metadata, hidden_dim, n_layers, bidirectional):
        super().__init__()

        if USE_STATEMENT_BRANCH:
            self.textcnn = TextCNN(vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, padding_idx)
        if USE_PRETRAINED_JUSTIFICATION:
            self.justification_encoder = DistilBertJustification(output_dim, dropout)
        else:
            self.justification_cnn = TextCNN(vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, padding_idx)
        self.textcnn2 = TextCNN(vocab_size, input_dim, n_filters, filter_sizes, output_dim, dropout, padding_idx)
        self.cnn_bilstm = CNNBiLSTM(input_dim_metadata, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout)
        self.fuzzy = FuzzyLayer(output_dim, output_dim)
        branch_count = 4 + int(USE_STATEMENT_BRANCH)
        self.fuse = nn.Linear(output_dim * branch_count, output_dim)
    
    def forward(self, text, metadata_text, metadata_number, justification, justification_attention_mask):
        #text = [batch size, sent len]
        #metadata = [batch size, metadata dim]

        branch_outputs = []
        if USE_STATEMENT_BRANCH:
            branch_outputs.append(self.textcnn(text))
        metadata_output_text = self.textcnn2(metadata_text)
        metadata_output_number = self.cnn_bilstm(metadata_number)
        metadata_output_fuzzy = self.fuzzy(metadata_output_number)
        if USE_PRETRAINED_JUSTIFICATION:
            justification_output = self.justification_encoder(justification, justification_attention_mask)
        else:
            justification_output = self.justification_cnn(justification)

        branch_outputs.extend(
            (metadata_output_text, metadata_output_number, metadata_output_fuzzy, justification_output)
        )
        fused_output = self.fuse(torch.cat(branch_outputs, dim=1))

        return fused_output


vocab_size = 30522
embedding_dim = 128
n_filters = 128
filter_sizes = [3,4,5]
output_dim = 6
dropout = 0.5
padding_idx = 0
input_dim = 6 * metadata_each_dim
input_dim_metadata = NUMERIC_FEATURE_DIM
hidden_dim = 64
n_layers = 1
bidirectional = True

class ClassificationLoss(nn.Module):
    def __init__(self, label_smoothing=0.0, ordinal_weight=0.0):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.ordinal_weight = ordinal_weight

    def forward(self, logits, targets):
        labels = targets.argmax(dim=1) if targets.ndim == 2 else targets.long()
        cross_entropy = F.cross_entropy(
            logits,
            labels,
            label_smoothing=self.label_smoothing
        )
        if self.ordinal_weight == 0:
            return cross_entropy

        probabilities = torch.softmax(logits, dim=1)
        class_ids = torch.arange(logits.size(1), device=logits.device)
        squared_distances = (class_ids.unsqueeze(0) - labels.unsqueeze(1)).square().float()
        ordinal_penalty = (probabilities * squared_distances).sum(dim=1).mean()
        return cross_entropy + self.ordinal_weight * ordinal_penalty / (logits.size(1) - 1) ** 2


def build_model(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return LiarModel(
        vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout,
        padding_idx, input_dim, input_dim_metadata, hidden_dim, n_layers, bidirectional
    ).to(DEVICE)


def build_model_and_optimizer(seed):
    model = build_model(seed)
    if USE_PRETRAINED_JUSTIFICATION:
        encoder_params = list(model.justification_encoder.encoder.parameters())
        encoder_param_ids = {id(param) for param in encoder_params}
        classifier_params = [param for param in model.parameters() if id(param) not in encoder_param_ids]
        optimizer = torch.optim.AdamW(
            [
                {'params': encoder_params, 'lr': encoder_learning_rate},
                {'params': classifier_params, 'lr': learning_rate},
            ],
            weight_decay=0.01
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer


criterion = (
    ClassificationLoss(LABEL_SMOOTHING, ORDINAL_LOSS_WEIGHT)
    if USE_PRETRAINED_JUSTIFICATION or LABEL_SMOOTHING > 0 or ORDINAL_LOSS_WEIGHT > 0
    else nn.BCEWithLogitsLoss()
)


# Record the training process
Train_acc = []
Train_loss = []
Train_macro_f1 = []
Train_micro_f1 = []

Val_acc = []
Val_loss = []
Val_macro_f1 = []
Val_micro_f1 = []

def train(num_epochs, model, train_loader, val_loader, optimizer, criterion, model_save):
    epoch_trained = 0
    best_valid_loss = float('inf')
    best_valid_accuracy = 0.0
    epochs_without_improvement = 0

    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_trained += 1
        epoch_start_time = time.time()
        train_label_all = []
        train_predict_all = []
        val_label_all = []
        val_predict_all = []
        # Training
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        for statements, label_onehot, label, metadata_text, metadata_number, justification, justification_attention_mask in train_loader:
            statements = statements.to(DEVICE)
            label_onehot = label_onehot.to(DEVICE)
            label = label.to(DEVICE)
            metadata_text = metadata_text.to(DEVICE)
            metadata_number = metadata_number.to(DEVICE)
            justification = justification.to(DEVICE)
            justification_attention_mask = justification_attention_mask.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(statements, metadata_text, metadata_number, justification, justification_attention_mask)
            loss = criterion(outputs, label_onehot)
            loss.backward()
            if USE_PRETRAINED_JUSTIFICATION:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            _, train_predicted = torch.max(outputs, 1)
            train_accuracy += sum(train_predicted == label)
            train_predict_all += train_predicted.tolist()
            train_label_all += label.tolist()
        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader.dataset)
        train_macro_f1 = f1_score(train_label_all, train_predict_all, average='macro')
        train_micro_f1 = f1_score(train_label_all, train_predict_all, average='micro')

        Train_acc.append(train_accuracy.tolist())
        Train_loss.append(train_loss)
        Train_macro_f1.append(train_macro_f1)
        Train_micro_f1.append(train_micro_f1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        with torch.no_grad():
            for statements, label_onehot, label, metadata_text, metadata_number, justification, justification_attention_mask in val_loader:
                statements = statements.to(DEVICE)
                label_onehot = label_onehot.to(DEVICE)
                label = label.to(DEVICE)
                metadata_text = metadata_text.to(DEVICE)
                metadata_number = metadata_number.to(DEVICE)
                justification = justification.to(DEVICE)
                justification_attention_mask = justification_attention_mask.to(DEVICE)

                val_outputs = model(statements, metadata_text, metadata_number, justification, justification_attention_mask)
                val_loss += criterion(val_outputs, label_onehot).item()
                _, val_predicted = torch.max(val_outputs, 1)
                val_accuracy += sum(val_predicted == label)
                val_predict_all += val_predicted.tolist()
                val_label_all += label.tolist()
        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader.dataset)
        val_macro_f1 = f1_score(val_label_all, val_predict_all, average='macro')
        val_micro_f1 = f1_score(val_label_all, val_predict_all, average='micro')

        Val_acc.append(val_accuracy.tolist())
        Val_loss.append(val_loss)
        Val_macro_f1.append(val_macro_f1)
        Val_micro_f1.append(val_micro_f1)

        if USE_PRETRAINED_JUSTIFICATION:
            is_best = val_accuracy > best_valid_accuracy
        else:
            is_best = val_loss < best_valid_loss

        if is_best:
            best_valid_loss = val_loss
            best_valid_accuracy = val_accuracy
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_save)
            print(f'***** Best Result Updated at Epoch {epoch_trained}, Val Acc: {val_accuracy:.4f}, Val Loss: {val_loss:.4f} *****')
        elif USE_PRETRAINED_JUSTIFICATION:
            epochs_without_improvement += 1

        # Print the losses and accuracy
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        print(f"Epoch [{epoch+1}/{num_epochs}], Time: {epoch_time:.2f}s, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Train F1 Macro: {train_macro_f1:.4f}, Train F1 Micro: {train_micro_f1:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val F1 Macro: {val_macro_f1:.4f}, Val F1 Micro: {val_micro_f1:.4f}")

        if USE_PRETRAINED_JUSTIFICATION and epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping after {early_stopping_patience} epochs without validation-accuracy improvement.")
            break

    end_time = time.time()
    training_time = end_time - start_time
    print(f'Total Training Time: {training_time:.2f}s')


member_checkpoint_paths = []
for member_index in range(ENSEMBLE_SIZE):
    member_seed = ENSEMBLE_SEED + member_index
    member_path = model_save.replace('.pt', f'_seed{member_seed}.pt') if ENSEMBLE_SIZE > 1 else model_save
    model, optimizer = build_model_and_optimizer(member_seed)
    print(f'\nTraining ensemble member {member_index + 1}/{ENSEMBLE_SIZE} (seed={member_seed})')
    train(num_epochs, model, train_loader, val_loader, optimizer, criterion, member_path)
    member_checkpoint_paths.append(member_path)
    del optimizer
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_ensemble(loader, split_name):
    member_probabilities = []
    expected_labels = None
    for member_path in member_checkpoint_paths:
        member_model = build_model(ENSEMBLE_SEED)
        member_model.load_state_dict(torch.load(member_path, map_location=DEVICE))
        member_model.eval()
        probabilities = []
        labels = []
        with torch.no_grad():
            for statements, _, label, metadata_text, metadata_number, justification, justification_attention_mask in loader:
                logits = member_model(
                    statements.to(DEVICE),
                    metadata_text.to(DEVICE),
                    metadata_number.to(DEVICE),
                    justification.to(DEVICE),
                    justification_attention_mask.to(DEVICE)
                )
                probabilities.append(torch.softmax(logits, dim=1).cpu())
                labels.append(label)
        member_probabilities.append(torch.cat(probabilities, dim=0))
        current_labels = torch.cat(labels, dim=0)
        if expected_labels is None:
            expected_labels = current_labels
        elif not torch.equal(expected_labels, current_labels):
            raise RuntimeError(f'{split_name} labels changed between ensemble passes.')
        del member_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_probabilities = torch.stack(member_probabilities).mean(dim=0)
    predictions = mean_probabilities.argmax(dim=1).numpy()
    labels = expected_labels.numpy()
    accuracy = float(np.mean(predictions == labels))
    macro_f1 = f1_score(labels, predictions, average='macro')
    micro_f1 = f1_score(labels, predictions, average='micro')
    print(f'{split_name} ensemble (n={ENSEMBLE_SIZE}) accuracy={accuracy:.4f} '
          f'f1_macro={macro_f1:.4f} f1_micro={micro_f1:.4f}')
    return accuracy


if ENSEMBLE_SIZE > 1:
    evaluate_ensemble(val_loader, 'Validation')
evaluate_ensemble(test_loader, 'Test')