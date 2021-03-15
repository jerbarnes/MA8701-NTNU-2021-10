import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import SentimentDataset, Vocab
from lstm import LSTM_Model
from sklearn.metrics import f1_score

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from time import time

import argparse

def svm_predict_text(svm_model, vectorizer, text):
    return svm_model.predict(vectorizer.transform([text]))[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embedding_dim", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=50)
    parser.add_argument("--training_data", default="data/train_small.tsv")
    # setup hyperparameters
    args = parser.parse_args()

    # Load the data and create the vocabulary at the same time
    # The Vocab object is a default dictionary that creates a new entry
    # for each new word that it sees while loading the datasets
    vocab = Vocab()
    train_data = SentimentDataset(args.training_data, vocab).get_split()
    dev_data = SentimentDataset("data/dev.tsv", vocab).get_split()
    test_data = [l.strip().split("\t") for l in open("data/test.tsv")]
    vocab.eval()

    ###################################################################
    # BOW SETUP
    ###################################################################

    # we create a sklearn vectorizer which is initialized with the same
    # vocabulary as we will use for the LSTM, making the comparison more fair
    vectorizer = CountVectorizer(vocabulary=vocab)
    train_y, train_text = zip(*[l.strip().split("\t") for l in open(args.training_data)])
    bow_train = vectorizer.transform(train_text)
    dev_y, dev_text = zip(*[l.strip().split("\t") for l in open("data/dev.tsv")])
    bow_dev = vectorizer.transform(dev_text)
    test_y, test_text = zip(*[l.strip().split("\t") for l in open("data/test.tsv")])
    bow_test = vectorizer.transform(test_text)

    print("Training Linear SVM on Bag of Words representations...")
    svm_model = LinearSVC()
    start_time = time()
    svm_model.fit(bow_train, train_y)
    end_time = time()

    print("{0:.1f} seconds to train".format(end_time - start_time))
    print()

    pred = svm_model.predict(bow_train)
    score = f1_score(train_y, pred, average="macro")
    print("Train F1: {0:.3f}".format(score), end="\t")
    pred = svm_model.predict(bow_dev)
    score = f1_score(dev_y, pred, average="macro")
    print("Dev F1: {0:.3f}".format(score))
    print()

    ###################################################################
    # LSTM SETUP
    ###################################################################
    print("Training LSTM on {} examples...".format(len(train_data)))
    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              collate_fn=train_data.collate_fn,
                              shuffle=True)

    dev_loader = DataLoader(dev_data,
                            batch_size=1,
                            collate_fn=dev_data.collate_fn,
                            shuffle=False)

    model = LSTM_Model(word2idx=vocab,
                       embedding_dim=args.embedding_dim,
                       hidden_dim=args.hidden_dim,
                       num_labels=2)

    # use cross entropy as the loss function
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # For each epoch in training epochs
    start_time = time()
    for i, epoch in enumerate(range(args.training_epochs)):
        model.train()
        print("Epoch {0}    ".format(i), end="\t")
        epoch_loss = 0
        for texts, labels in train_loader:
            # get lstm prediction
            pred = model.max_pool(texts)
            # calculate loss
            loss = loss_function(pred, labels.flatten())
            epoch_loss += loss.data
            # perform backwards gradient steps and update parameters
            loss.backward()
            optimizer.step()
            model.zero_grad()
        print("Loss: {0:.3f}".format(loss / args.training_epochs), end="\t")

        # check results on train
        train_preds, train_gold = [], []
        model.eval()
        for texts, labels in train_loader:
            # get lstm prediction
            logits = model.max_pool(texts)
            _, preds = logits.max(1)
            train_preds.extend(preds.numpy())
            train_gold.extend(labels.flatten().numpy())
        score = f1_score(train_gold, train_preds, average="macro")
        print("Train F1: {0:.3f}".format(score), end="\t\t")

        # check results on dev
        dev_preds, dev_gold = [], []
        model.eval()
        for texts, labels in dev_loader:
            # get lstm prediction
            logits = model.max_pool(texts)
            _, preds = logits.max(1)
            dev_preds.extend(preds.numpy())
            dev_gold.extend(labels.flatten().numpy())
        score = f1_score(dev_gold, dev_preds, average="macro")
        print("Dev F1: {0:.3f}".format(score))
    end_time = time()

    print("{0:.1f} seconds to train".format(end_time - start_time))
    print()

    model.eval()


    # Now we can evaluate both models on the test data
    print()
    print("Evaluating on the challenge data...")
    print()
    print("ID     \tGold     \tBOW Prediction\tLSTM Prediction\tText")
    print("-------\t---------\t--------------\t---------------\t----")
    for i, (label, text) in enumerate(test_data):
        svm_pred = svm_predict_text(svm_model, vectorizer, text)
        lstm_pred = model.predict_text(text)
        print("{0}\t{1}\t{2}\t{3}\t{4}".format(i, label, svm_pred, lstm_pred, text))
