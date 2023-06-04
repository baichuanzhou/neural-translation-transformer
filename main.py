from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List, Union, Any
import torch
import torch.nn as nn
from transformer import EncoderDecoder
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from timeit import default_timer as timer
import random

torch.manual_seed(42)

multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training" \
                        ".tar.gz"
multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation" \
                        ".tar.gz"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 20

SRC_LANGUAGE = "de"  # source language is german
TGT_LANGUAGE = "en"  # our target is to translate german to english

token_transform = {}
vocab_transform = {}

token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')


def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}
    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])


# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # Training data Iterator
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    # Create torchtext's Vocab object
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

# Set ``UNK_IDX`` as the default index. This index is returned when the token is not found.
# If not set, it throws ``RuntimeError`` when the queried token is not found in the Vocabulary.
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[ln].set_default_index(UNK_IDX)


# create masks to prevent transformer to look into the future words
def subsequent_mask(sz: int) -> torch.Tensor:
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.size(0)
    tgt_seq_len = tgt.size(0)

    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)
    tgt_mask = subsequent_mask(tgt_seq_len)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask.to(DEVICE), tgt_mask.to(DEVICE), src_padding_mask.to(DEVICE), tgt_padding_mask.to(DEVICE)


"""
    Here define the parameters for transformer
"""
SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
D_MODEL = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3


def adjust_learning_rate(optimizer, d_model, step_num, warmup_steps=3000):
    lr = d_model ** -0.5 * min(step_num ** -0.5, step_num * warmup_steps ** -1.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func


def tensor_transform(tensor_ids: List[int]):
    return torch.cat((
        torch.tensor([BOS_IDX]),
        torch.tensor(tensor_ids),
        torch.tensor([EOS_IDX])
    ))


# ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln],  # Tokenization
                                               vocab_transform[ln],  # Numericalization
                                               tensor_transform)  # Add BOS/EOS and create tensor


# function to collate data samples into batch tensors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


class TrainState:
    steps: int = 0   # total steps taken
    samples: int = 0    # samples used
    tokens: int = 0     # tokens processed


def train_epoch(model: nn.Module, optimizer: torch.optim, criterion: nn.Module, train_state: TrainState):
    model.train()
    model = model.to(DEVICE)
    losses = 0
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    train_loader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)

    for src, tgt in train_loader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        train_state.steps += 1

        optimizer.step()
        losses += loss.item()

        train_state.samples += src.size(0)
        train_state.tokens += (tgt != PAD_IDX).data.sum()

    return losses / len(list(train_loader))


def evaluate(model: nn.Module, criterion: nn.Module):
    losses = 0

    val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    val_loader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    model.eval()
    with torch.no_grad():
        for src, tgt in val_loader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            tgt_input = tgt[:-1, :]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
            logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

            tgt_out = tgt[1:, :]
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()
    return losses / len(list(val_loader))


def greedy_decode(model, src, src_mask, max_len, start_symbol) -> torch.Tensor:
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len - 1):
        memory = memory.to(DEVICE)
        tgt_mask = subsequent_mask(ys.size(0)).type(torch.bool).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)

        if next_word == EOS_IDX:
            break
    return ys


def translate(model: nn.Module, src_sentence: str):
    model.eval()
    with torch.no_grad():
        src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
        num_tokens = src.size(0)
        src_mask = torch.zeros(num_tokens, num_tokens).type(torch.bool)
        tgt_tokens = greedy_decode(model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
        return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))) \
            .replace("<bos>", "").replace("<eos>", "")


def train(model: nn.Module,
          optimizer: optim,
          criterion: Union[nn.Module, Any]
          ):
    model.train()
    model = model.to(DEVICE)
    train_state = TrainState()
    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = timer()
        train_loss = train_epoch(model, optimizer, criterion, train_state)
        end_time = timer()

        val_loss = evaluate(model, criterion)
        print(f"Epoch: {epoch}, Train loss: {train_loss: .3f}, Val loss: {val_loss: .3f}, "
              f""f"Epoch time = {(end_time - start_time): .3f}s, Steps: {train_state.steps}, "
              f"LR: {optimizer.param_groups[0]['lr']: .6f}, Tokens Processed: {train_state.tokens}, "
              f"Samples Seen: {train_state.samples}")
        random_index = random.randint(0, len(list(train_iter)))
        src_sentence, tgt_sentence = (list(train_iter))[random_index]
        translate_sentence = translate(model, src_sentence)
        print('> Input    Language:', src_sentence)
        print('= True  Translation:', tgt_sentence)
        print('< Model Translation:', translate_sentence)


def main():
    transformer = EncoderDecoder(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, D_MODEL,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
    transformer = transformer.to(DEVICE)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    train(transformer,
          optimizer,
          criterion)
    print(translate(transformer, "Eine Gruppe von Menschen steht vor einem Iglu ."))


if __name__ == '__main__':
    main()
