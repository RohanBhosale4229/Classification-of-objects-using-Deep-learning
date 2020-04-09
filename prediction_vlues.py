

from fastai.vision import*
import numpy as np

class MyLoss(nn.Module):
    def forward(self, yhat, bbox_tgts, class_tgts):
        det_loss=nn.L1Loss()(yhat[:,:4].unsqueeze_(dim=1), bbox_tgts)
        cls_loss=nn.CrossEntropyLoss()(yhat[:,4:], class_tgts.view(-1))
        #print(det_loss, cls_loss)
        
        return det_loss + 1.0*cls_loss



class Model(object):
    def __init__(self, path='/content/drive/My Drive/Colab Notebooks/bbc_train', file='export.pkl'):
        
        self.learn=load_learner(path=path, file=file) #Load model
        self.class_names=['brick', 'ball', 'cylinder'] #Be careful here, labeled data uses this order, but fastai will use alphabetical by default!
    

    def predict(self, x):
       

        #Normalize input data using the same mean and std used in training:
       
        #x_norm=normalize(x, torch.tensor(self.learn.data.stats[0]), 
         #                   torch.tensor(self.learn.data.stats[1]))
    

        #Pass data into model:
        def compute_corner_locations(y, im_shape=(256,256)):
          shape_vec=np.array(im_shape*2)
          bounds=((y+1)*shape_vec/2).ravel()
          corners=np.array([bounds[0], bounds[1],bounds[2],bounds[3]])
          return corners
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            yhat=self.learn.model(x.to(device))
            yhat=yhat.detach().cpu()

        pred_class=yhat[:,4:].argmax(1)
        
        pred_class=[self.learn.data.classes[i] for i in pred_class]
        
        li=[]

        for i in range(64): 
          li.append(compute_corner_locations(yhat[i][:4].cpu().numpy()))
      
        
        #Post-processing/parsing outputs, here's an example for classification only:
        '''
        class_prediction_indices=yhat.argmax(dim=1)
        class_predictions=[self.learn.data.classes[i] for i in class_prediction_indices]

        #Random Selection Placeholder Code for testing
        #class_predictions=[self.class_names[np.random.randint(3)] for i in range(x.shape[0])]
      
        #Scale randomly chosen bbox coords to image shape:
        bbox=np.random.rand(x.shape[0], 1)
        bbox[:,0] *= x.shape[2]; #bbox[:,2] *= x.shape[2] 
        bbox[:,0] *= x.shape[3]; #bbox[:,3] *= x.shape[3]
        
        '''
        #Create random segmentation mask:
        mask=np.random.randint(low=0, high=1, size=(x.shape[0], x.shape[2], x.shape[3]))
       
        return (pred_class, li, mask)






