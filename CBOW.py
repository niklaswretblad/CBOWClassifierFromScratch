

import numpy as np


def load_data(filename, max_length=20):
    items = []
    with open(filename, 'rt', encoding='utf-8') as fp:
        for line in fp:
            sentence, label = line.rstrip().split('\t')
            items.append((sentence.split()[:max_length], int(label)))
    return items


def make_vocab(data):
    vocab = {'<pad>': 0}
    for sentence, label in data:
        for t in sentence:
            if t not in vocab:
                vocab[t] = len(vocab)
    return vocab


def vectorize(vocab, data):
    max_length = max(len(s) for s, _ in data)
    xs = []
    ys = []
    for sentence, label in data:
        x = [0] * max_length
        for i, w in enumerate(sentence):
            if w in vocab:
                x[i] = vocab[w]
        xs.append(x)
        ys.append(label)
    return np.array(xs), np.array(ys)


def accuracy(y_pred, y):
    return np.mean((y_pred == y).astype(float))


def minibatches(x, y, batch_size):
    random_indices = np.random.permutation(x.shape[0])
    for i in range(0, x.shape[0] - batch_size + 1, batch_size):
        batch_indices = random_indices[i:i+batch_size]
        yield x[batch_indices], y[batch_indices]


# Efficient implementation of the softmax function; see
# http://stackoverflow.com/a/39558290

def softmax(X):
    E = np.exp(X - np.max(X, axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)


class CBOW(object):

    def __init__(self, num_embeddings, output_dim, embedding_dim=30):
        k = 1/embedding_dim
        self.E = np.random.normal(0, 0.1, (num_embeddings, embedding_dim))
        self.W = np.random.uniform(low=-np.sqrt(k), high=np.sqrt(k), size=(embedding_dim, output_dim))
        self.b = np.random.uniform(low=-np.sqrt(k), high=np.sqrt(k), size=(1, output_dim))

    def forward(self, features):    
        embeddings = self.E[(features.flatten()), ]        
        embeddings = embeddings.reshape((features.shape[0], features.shape[1], self.E.shape[1]))
        
        return embeddings.mean(axis=1) @ self.W + self.b
    
    def backward(self, features, error, lr):        
        B = features.shape[0]
        L = features.shape[1]
        embeddings = self.E[(features.flatten()), ]
        embeddings = embeddings.reshape(features.shape[0], features.shape[1], self.E.shape[1])
        E_mean = embeddings.mean(axis=1)
        
        self.W = self.W - (lr / B) * (E_mean.T @ error)
        self.b = self.b - (lr / B) * (np.ones((B, 1)).T @ error)

        E_error = (1 / L) * (error @ self.W.T)
        
        for batch in range(E_error.shape[0]):
            self.E[features[batch], ] -= E_error[batch]


def train(vocab, train_x, train_y, n_epochs=20, batch_size=24, lr=1e-1):
    n_classes = len(set(train_y))
    
    model = CBOW(len(vocab), n_classes, embedding_dim=30)
    
    for e in range(n_epochs):
        print("Training epoch: ", e)
        for x, y in minibatches(train_x, train_y, batch_size):
            output = model.forward(x)
            error = softmax(output)            
            error[np.arange(len(y)), y] -= 1            
            model.backward(x, error, lr)            

    return model


# MAIN METHOD


def main():
    import sys

    LR = 1e-2
    BATCH_SIZE = 24
    EPOCHS = 20

    train_data = load_data('sst-5-train.txt')
    dev_data = load_data('sst-5-dev.txt')

    vocab = make_vocab(train_data)
    train_x, train_y = vectorize(vocab, train_data)
    dev_x, dev_y = vectorize(vocab, dev_data)    

    accuracies = []
    for _ in range(10):
        model = train(vocab, train_x, train_y, n_epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR)
        dev_output = model.forward(dev_x)
        dev_y_pred = np.argmax(dev_output, axis=1)
        accuracies.append(accuracy(dev_y_pred, dev_y))
    print('Average accuracy: {:.4f}'.format(sum(accuracies) / len(accuracies)))


if __name__ == '__main__':
    main()
