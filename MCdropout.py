import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import time

class MCdrop(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=6, nu=1, lamdba=1,N=20):
        super(MCdrop, self).__init__()
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
        self.dropout = torch.nn.Dropout(0.1)
        print(self.dropout)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax()
        self.xent_func = torch.nn.BCELoss()
        self.N = N

    def forward(self, x_user, x_item):
        U_emb = self.W(x_user)
        V_emb = self.H(F.one_hot(x_item, num_classes=self.num_items).float())
        z_emb = torch.cat([U_emb, V_emb], axis=1)
        h1 = self.linear_1(z_emb)
        h1 = self.relu(h1)
        h2 = self.linear_2(h1)
        h2 = self.relu(h2)
        h2 = torch.cat((h2, torch.Tensor(np.ones((h2.size()[0],1))).cuda()),1)
        h3 = self.dropout(h2)
        out = self.linear_3(h3)
        return out, h3

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
                pred, grad = self.forward(sub_x_user,sub_x_item)
                pred = self.sigmoid(pred)
                loss = self.xent_func(pred.float(), torch.unsqueeze(sub_y.float(),1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.cpu().detach().numpy()                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            last_loss = epoch_loss
        
    def predict(self, x_user, x_item):
        with torch.no_grad():
            st0 = time.time()
            starttime = time.time()
            if not torch.is_tensor(x_user):
                x_user = torch.Tensor(x_user).cuda()
            if not torch.is_tensor(x_item):
                x_item = torch.LongTensor(x_item).cuda()
            ed0 = time.time()
            dt0 = ed0 - st0
            #print("num of forward is :",self.N,";time cost0 is:",dt0)
            pred_all = []
            
            st1 = time.time()
            for i in range(self.N):
                pred, grad = self.forward(x_user, x_item)
                #print(pred.shape)
                pred_all.append(pred)
            ed1 = time.time()
            dt1=ed1-st1
            #print("time cost 1:",dt1)

            st2 = time.time()
            mean = sum(pred_all)/self.N     
            for i in range(self.N):
                pred_all[i] = (pred_all[i]-mean)*(pred_all[i]-mean)
            #torch.concat
            #for i in range(self.N):
            #    pred_all[i]=pred_all[i]*pred_all[i]
            sigma = torch.sqrt(sum(pred_all)/(self.N-1))
            #print(torch.mean(sigma))
            #print(sigma.shape)
            ed2 = time.time()
            dt2=ed2-st2
            #print("time cost 2:",dt2)

            #tmp = torch.sqrt(torch.sum(self.lamdba * self.nu * grad * grad / self.U, dim=1))
            #sigma =torch.ones_like(tmp)*0.2
            #print(sigma)
            st3 = time.time()
            pred = torch.normal(mean.view(-1), sigma.view(-1))
            pred = self.sigmoid(pred)
            #print("max1:",max(pred))
            res_pred = pred.cpu().detach().numpy().flatten()
            ed3 = time.time()
            dt3=ed3-st3
            #print("time cost 3:",dt3)
        return res_pred