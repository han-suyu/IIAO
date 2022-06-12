
from torch.utils.data import DataLoader
import torch
import numpy as np
import argparse
from net import IIAO
from dataset import IIAODataset




def get_args_parser():
    parser = argparse.ArgumentParser('Inference setting', add_help=False)
    parser.add_argument('--level', type=str,default='')
    parser.add_argument('--mode', type=str,default='')
    return parser



def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define data
    IMG_DIR = 'datasets/jhu_crowd_v2.0/'+args.level+'/'+args.mode+'/images' 
    val_dataset = IIAODataset(IMG_DIR,  scale = 8)
    data_loader = DataLoader(val_dataset, batch_size = 1, num_workers=0)



    # define model
    model = IIAO().float()
    model = model.to(device)

    # load the trained model
    state = torch.load('checkpoint/Overall-'+args.mode+'.pth')
    model.load_state_dict(state['net'])


    
        
        
    # test mode
    model.eval()

    mae, mse = 0.0, 0.0
    cnt = 0


    with torch.no_grad():
        for data in data_loader:
            cnt += 1


            image, gt,path = data
            print(path[0],'   ',cnt,' ===> ',len(data_loader))

            image = image.float()
            image = image.to(device)
        

            _,_,pr_density = model(image)


            pr_density = pr_density.cpu().detach().numpy()
            pr = np.sum(pr_density)
        
            mae += np.abs(gt - pr)
            mse += np.abs(gt - pr) ** 2

            # print(path[0],'  loss:', np.abs(gt - pr).item(),'  mae:', (mae/cnt).item(),'  mse:', (np.sqrt(mse/cnt)).item())

        


    # calculate loss
    final_mae = mae / cnt
    final_mse = np.sqrt(mse / cnt)



    print('mae:',final_mae.item())
    print('mse:',final_mse.item())



if __name__ == '__main__':


    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    print(args.__dict__)
 
    main(args)
