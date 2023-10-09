import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import time

class NeuralTS(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=6, nu=1, lamdba=1):
        super(NeuralTS, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.nu = nu
        self.lamdba = lamdba
        self.W = torch.nn.Linear(self.num_users, self.embedding_k, bias=False).cuda()
        self.H = torch.nn.Linear(self.num_items, self.embedding_k, bias=False).cuda()
        self.linear_1 = torch.nn.Linear(12, 16).cuda()
        self.linear_2 = torch.nn.Linear(16, 8).cuda()
        self.linear_3 = torch.nn.Linear(9, 1, bias=False).cuda()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax()
        self.xent_func = torch.nn.BCELoss()
        self.param = list(self.parameters())
        self.U = lamdba * torch.tensor(np.ones(5087)).cuda()

    def forward(self, x_user, x_item):
        U_emb = self.W(x_user)
        V_emb = self.H(F.one_hot(x_item, num_classes=self.num_items).float())
        z_emb = torch.cat([U_emb, V_emb], axis=1)
        h1 = self.linear_1(z_emb)
        h1 = self.relu(h1)
        h2 = self.linear_2(h1)
        h2 = self.relu(h2)
        h2 = torch.cat((h2, torch.Tensor(np.ones((h2.size()[0],1))).cuda()),1)
        out = self.linear_3(h2)
        return out

    def fit(self, x_user, x_item, y, num_epoch=1, lamb=0, lr=0.01, batch_size=64):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9
        num_sample = len(x_user)
        total_batch = num_sample // batch_size
        early_stop = 0

        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            epoch_loss = 0
            for idx in range(total_batch):
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x_user = x_user[selected_idx]
                sub_x_item = x_item[selected_idx]
                sub_y = y[selected_idx]
                optimizer.zero_grad()
                if not torch.is_tensor(sub_x_user):
                    sub_x_user = torch.Tensor(sub_x_user).cuda()
                if not torch.is_tensor(sub_x_item):
                    sub_x_item = torch.LongTensor(sub_x_item).cuda()
                if not torch.is_tensor(sub_y):
                    sub_y = torch.Tensor(sub_y).cuda()
                pred = self.forward(sub_x_user,sub_x_item)
                pred = self.sigmoid(pred)
                loss = self.xent_func(pred.float(), torch.unsqueeze(sub_y.float(),1))
                #print(loss.shape)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.cpu().detach().numpy()                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            last_loss = epoch_loss

        all_idx = np.arange(num_sample)
        for idx in range(total_batch):
            selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
            sub_x_user = x_user[selected_idx]
            sub_x_item = x_item[selected_idx]
            sub_y = y[selected_idx]
            optimizer.zero_grad()
            if not torch.is_tensor(sub_x_user):
                sub_x_user = torch.Tensor(sub_x_user).cuda()
            if not torch.is_tensor(sub_x_item):
                sub_x_item = torch.LongTensor(sub_x_item).cuda()
            if not torch.is_tensor(sub_y):
                sub_y = torch.Tensor(sub_y).cuda()
            with torch.no_grad():
                pred = self.forward(sub_x_user,sub_x_item)
            #self.U += torch.sum(grad*grad, dim=0) 

    def grad(self, x):
        y = self.net(x)
        self.optimizer.zero_grad()
        y.backward()
        return torch.cat(
                [w.grad.detach().flatten() / np.sqrt(self.hidden_size) for w in self.net.parameters() if w.requires_grad]
            ).to(self.device)
      
    def predict(self, x_user, x_item):
        
        #with torch.no_grad():
        if not torch.is_tensor(x_user):
            x_user = torch.Tensor(x_user).cuda()
        if not torch.is_tensor(x_item):
            x_item = torch.LongTensor(x_item).cuda()
        #starttime0 = time.time()
        pred = self.forward(x_user, x_item)
        #endtime0 = time.time()
        #dtime0= endtime0-starttime0
        #print("time cost0 is:",dtime0)
        num_sample = pred.shape[0]

        #print(num_sample)
        final_pred=[]
        final_sigma=[]
        sigma_mean = 0
        #starttime1 = time.time()
        for i in range(num_sample):
            starttime2 = time.time()
            pred[i].backward(retain_graph=True)
            endtime2 = time.time()
            dtime2 =endtime2-starttime2
            #print("time cost backward is:",dtime2)

            #grad = [x.grad.view(-1) for x in self.param]
            #final_grad = torch.cat(grad,dim=0)
            #if i == 0:
            #   print(final_grad.shape)
            self.zero_grad() 
            #sigma = torch.sqrt(torch.sum(self.lamdba * self.nu * final_grad * final_grad/450))
            #sigma_mean += sigma
            #print(sigma.shape)
            #pred_tmp = torch.normal(pred[i].view(-1), sigma.view(-1))
            #print(pred_tmp.shape)
            #print(pred_tmp)
            #final_pred.append(pred_tmp)
            #tmp = torch.sqrt(torch.sum(self.lamdba * self.nu * grad * grad / self.U, dim=1))
        #endtime1 = time.time()
        #dtime1=endtime1-starttime1
        #print("time cost2 is:",dtime2)
        sigma_mean=sigma_mean/num_sample
        print("sigma_mean is :",sigma_mean)
        #print(final_pred)
        #print("___________________________")
        final_pred = torch.tensor(final_pred)
        final_pred = self.sigmoid(final_pred)
        #print(final_pred.shape)
        #print(type(final_pred))
        #print("___________________________")
        res_pred = final_pred.cpu().detach().numpy().flatten()
        print(res_pred.shape)
        return res_pred