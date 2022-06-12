import numpy as np
import os
import cv2
import scipy.io as scio


def generate_mat(f):
    base = './datasets/jhu_crowd_v2.0/'+f+'/'
 



    mat_folder = base+'mats/'
    # shutil.rmtree(mat_folder)
    if not os.path.isdir(mat_folder):
        os.mkdir(mat_folder)



    check_folder = base+'check_mat/'
    # shutil.rmtree(check_folder)
    if not os.path.isdir(check_folder):
        os.mkdir(check_folder)




    img_files = os.listdir(base+'images/')
    img_files.sort()

    txt_files = os.listdir(base+'gt/')
    txt_files.sort()




    for img_name,txt_name in zip(img_files, txt_files):
        img = cv2.imread(base+'images/'+img_name)

        gao,kuan = img.shape[0:2]

        mat=[]
        gt_count = 0
        error_count=0

        data = open(base+'gt/'+txt_name, 'r')
        for lines in data:
            x,y,w,h,_,_  = lines.strip().split(' ')
            x=int(x)
            y = int(y)
            if  x<kuan  and y<gao:
                cv2.circle(img,(x,y),2,(255,0,255),-1)
                mat.append([x,y])
                gt_count+=1
            else:
                error_count+=1
          
        print(img_name,error_count,' points rcoss the border...')
        # cv2.putText(img,'gt:'+str(gt_count), (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
        cv2.imwrite(check_folder+img_name, img)
        scio.savemat(mat_folder+img_name.replace('.jpg','.mat'),{'annPoints':mat,'count':gt_count})
     


if __name__ == '__main__':

    folder = ['train', 'val','test']

    for f in folder:
        generate_mat(f)
        print('******************************************************')
        print('Part  '+f+'  completed!!!')
        print('******************************************************')

   