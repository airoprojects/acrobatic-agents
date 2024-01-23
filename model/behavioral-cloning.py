###### Define student agent
import torch 
import torch.nn as nn
import torch.optim as optim


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class StudentAgent:

  def __init__(self, env, train_loader, test_loader, learning_rate):
    self.env = env
    self.train_loader = train_loader
    self.test_loader = test_loader

    n_inputs = env.observation_space.shape[0]
    n_outputs = env.action_space.n

    self.policy = nn.Sequential(
                  nn.Linear(n_inputs, 16),
                  nn.ReLU(),
                  nn.Linear(16, n_outputs),
                  nn.Softmax(dim=-1)
                )
    print("policy net: ", self.policy)

    self.loss_criterion = nn.CrossEntropyLoss()
    self.optimizer =  optim.Adam(self.policy.parameters(), lr=learning_rate)
    self.num_eval_episodes = 10

  def train(self, num_epochs, loader):
    self.policy.train()
    self.policy.to(device)

    for epoch in range(num_epochs):
      for batch_idx, (data, target) in enumerate(loader):
        obs, expert_action = data.to(device), target.to(device)
        
        self.optimizer.zero_grad()

        obs = obs.float()
        student_action = self.policy(obs)
        expert_action = expert_action.long()

        loss = self.loss_criterion(student_action, expert_action)
        loss.backward()
        self.optimizer.step()

      #compute accuracy
      train_acc = self.compute_accuracy(self.train_loader)
      test_acc = self.compute_accuracy(self.test_loader)
      
      print("Epoch {}:\ttrain accuracy: {}\ttest accuracy: {}".format(epoch, train_acc, test_acc))

  def compute_accuracy(self, loader):
    total = 0
    correct = 0

    self.policy.eval()
    test_loss = 0

    with torch.no_grad():
      for data, target in loader:
        obs, expert_action = data.to(device), target.to(device)

        obs = obs.float()
        student_action = self.policy_action(obs)

        total += student_action.size()[0]
        correct += sum(student_action==expert_action).item()

    accuracy = 100. * correct/(float)(total)

    return accuracy

