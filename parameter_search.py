import torch.optim as optim

lr = [0.1, 0.01, 0.001]
optimizers = ["adadelta", "adam", "sgd"]
momentum = [0, 0.5, 0.9]


def find_optimizer(lr=lr, optimizers=optimizers, model=None):
    for i in range(len(optimizers)):
        if optimizers[i] == "adadelta":
            for j in range(len(lr)):
                print(lr[j], "adad")
                yield optim.Adadelta(model.parameters(), lr=lr[j]), ("adadelta", str(lr[j]))
        elif optimizers[i] == "adam":
            for j in range(len(lr)):
                print(lr[j], "adam")
                yield optim.Adam(model.parameters(), lr=lr[j]), ("adam", str(lr[j]))

        elif optimizers[i]=="sgd":
            for j in range(len(lr)):
                for z in range(len(momentum)):
                    print(lr[j], momentum[z], "sgd")
                    yield optim.SGD(model.parameters(), lr=lr[j], momentum=momentum[z]), ("sgd", str(lr[j]), str(momentum[z]))
