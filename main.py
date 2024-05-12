import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from helper_functions import plot_decision_boundary
from multiclassModel import multiclassModel

# Set the hyperparameters for data creation
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

# 1. Create multi-class data
X_blob, y_blob = make_blobs(n_samples=1000,
    n_features=NUM_FEATURES, # X features
    centers=NUM_CLASSES, # y labels 
    cluster_std=1.5, # give the clusters a little shake up (try changing this to 1.0, the default)
    random_state=RANDOM_SEED
)

# 2. Turn data into tensors
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)


# 3. Split into train and test sets
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
    y_blob,
    test_size=0.2,
    random_state=RANDOM_SEED
)


# 4. Plot data
# plt.figure(figsize=(10, 7))
# plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
# plt.show()
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)

# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

model_0 = multiclassModel(hidden_units=10,output_features=4,input_features=2)
loss_fn = nn.CrossEntropyLoss();
optimizer = torch.optim.SGD(model_0.parameters(), 
                            lr=0.1) # exercise: try changing the learning rate here and seeing what happens to the model's performance


torch.cuda.manual_seed(42)
torch.manual_seed(42)

epochs = 10000
for epoch in range(epochs):
    model_0.train()

    # 1. Forward pass (model outputs raw logits)
    y_logits = model_0(X_blob_train) # squeeze to remove extra `1` dimensions, this won't work unless model and data are on same device 
    
    # y_pred = torch.round(torch.sigmoid(y_logits)) # turn logits -> pred probs -> pred labls
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

    # 2. Calculate loss/accuracy
    # loss = loss_fn(torch.sigmoid(y_logits), # Using nn.BCELoss you need torch.sigmoid()
    #                y_train) 

    loss = loss_fn(y_logits, # Using nn.BCEWithLogitsLoss works with raw logits
                   y_blob_train) 
    # print(y_logits.size())
    acc = accuracy_fn(y_true=y_blob_train, 
                      y_pred=y_pred) 

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model_0.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_0(X_blob_test)
        # test_pred = torch.round(torch.sigmoid(test_logits))
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
        # 2. Caculate loss/accuracy
        test_loss = loss_fn(test_logits,
                            y_blob_test)
        # print(y_blob_test.size())
        # print(test_pred.size())
        test_acc = accuracy_fn(y_true=y_blob_test,
                               y_pred=test_pred)

#     # Print out what's happening every 10 epochs
    # if epoch % 10 == 0:
    #     print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

# # Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, X_blob_train, y_blob_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_0, X_blob_test, y_blob_test)


# print(y_logits.size())
# print(y_pred.size())
print(y_blob_test.size())
print(test_pred.size())