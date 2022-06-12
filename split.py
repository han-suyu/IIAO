import imp
import numpy as np
import os
import cv2
import scipy.io as scio
import shutil
import h5py
import scipy.io as io


def new_folder():
    base = './datasets/jhu_crowd_v2.0/'

    folder1 = ['Overall','High','Medium','Low']
    folder2 = ['train','val','test']
    folder3 = ['density','images']


    for f1 in folder1:
        for f2 in folder2:
            for f3 in folder3:    
                path = base + f1+'/'+f2+'/'+f3
                if not os.path.exists(path):
                    os.makedirs(path)


def split(f):
    base = './datasets/jhu_crowd_v2.0/'




    h5_files = os.listdir(base+f+'/density/')
    h5_files.sort()

    low = 0
    medium = 0
    high = 0

    for name in h5_files:
        # mat = io.loadmat(base+f+'/density/'+name)
        # count = mat['count']
        with h5py.File(base+f+'/density/'+name, 'r') as hf:
            density = np.asarray(hf['density'])
        count = np.sum(density)

        
        if 0<=count<=50 :
            shutil.copyfile(base+f+'/density/'+name,   base+'Low/'+f+'/density/'+name)
            shutil.copyfile(base+f+'/images/'+name.replace('.h5','.jpg'),   base+'Low/'+f+'/images/'+name.replace('.h5','.jpg'))
            low+=1
        elif 50<count<=500 :
            shutil.copyfile(base+f+'/density/'+name,   base+'Medium/'+f+'/density/'+name)
            shutil.copyfile(base+f+'/images/'+name.replace('.h5','.jpg'),   base+'Medium/'+f+'/images/'+name.replace('.h5','.jpg'))
            medium+=1
        elif 500<count:
            shutil.copyfile(base+f+'/density/'+name,   base+'High/'+f+'/density/'+name)
            shutil.copyfile(base+f+'/images/'+name.replace('.h5','.jpg'),   base+'High/'+f+'/images/'+name.replace('.h5','.jpg'))
            high+=1
        shutil.copyfile(base+f+'/density/'+name,   base+'Overall/'+f+'/density/'+name)
        shutil.copyfile(base+f+'/images/'+name.replace('.h5','.jpg'),   base+'Overall/'+f+'/images/'+name.replace('.h5','.jpg'))
        
        
        print(f+ '_part  '+ name.replace('.h5','.jpg')+ ' || '+ str(low+medium+high)+'  =====>  '+str(len(h5_files)))


    assert low+medium+high==len(h5_files)




if __name__ == '__main__':

    new_folder()
    print('Successfully created multi-level folder!!!')


    folder = ['train', 'val','test']
    for f in folder:
        split(f)
        print('******************************************************')
        print('Part  '+f+'  completed!!!')
        print('******************************************************')

   
   
