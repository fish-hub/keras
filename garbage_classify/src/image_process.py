import os,shutil
from random import randint, sample
import random
i0,j0,k0,m0=0,0,0,0
i1,j1,k1,m1=0,0,0,0
basedir = "../垃圾分类"
testdir = "../test_picture"
train_base_dir = os.path.join(basedir,"train")
validation_base_dir = os.path.join(basedir,"validation")
train_metal_dir = os.path.join(train_base_dir,"metal")
train_glass_dir = os.path.join(train_base_dir,"glass")
train_paper_dir = os.path.join(train_base_dir,"paper")
train_plastic_dir = os.path.join(train_base_dir,"plastic")
validation_metal_dir = os.path.join(validation_base_dir,"metal")
validation_glass_dir = os.path.join(validation_base_dir,"glass")
validation_paper_dir = os.path.join(validation_base_dir,"paper")
validation_plastic_dir = os.path.join(validation_base_dir,"plastic")
os.mkdir(validation_base_dir)
os.mkdir(validation_metal_dir)
os.mkdir(validation_glass_dir)
os.mkdir(validation_paper_dir)
os.mkdir(validation_plastic_dir)
os.mkdir(testdir)

unfilter_fnames = os.listdir(train_glass_dir)
filter_fnames = sample(unfilter_fnames,66)
for fname in filter_fnames:
    src = os.path.join(train_glass_dir,fname)
    dst = os.path.join(validation_glass_dir,"glass{}.jpg".format(i0))
    shutil.move(src,dst)
    i0+=1
#分离验证集
unfilter_fnames = os.listdir(train_paper_dir)
filter_fnames = sample(unfilter_fnames,66)
for fname in filter_fnames:
    src = os.path.join(train_paper_dir,fname)
    dst = os.path.join(validation_paper_dir,"paper{}.jpg".format(j0))
    shutil.move(src,dst)
    j0+=1

unfilter_fnames = os.listdir(train_metal_dir)
filter_fnames = sample(unfilter_fnames,66)
for fname in filter_fnames:
    src = os.path.join(train_metal_dir,fname)
    dst = os.path.join(validation_metal_dir,"metal{}.jpg".format(k0))
    shutil.move(src,dst)
    k0+=1

unfilter_fnames = os.listdir(train_plastic_dir)
filter_fnames = sample(unfilter_fnames,66)
for fname in filter_fnames:
    src = os.path.join(train_plastic_dir,fname)
    dst = os.path.join(validation_plastic_dir,"plastic{}.jpg".format(m0))
    shutil.move(src,dst)
    m0+=1
j0,k0,m0= 0,0,0

#分离测试集
unfilter_fnames = os.listdir(train_paper_dir)
filter_fnames = sample(unfilter_fnames,4)
for fname in filter_fnames:
    src = os.path.join(train_paper_dir,fname)
    dst = os.path.join(testdir,"paper{}.jpg".format(j0))
    shutil.move(src,dst)
    j0+=1

unfilter_fnames = os.listdir(train_metal_dir)
filter_fnames = sample(unfilter_fnames,4)
for fname in filter_fnames:
    src = os.path.join(train_metal_dir,fname)
    dst = os.path.join(testdir,"metal{}.jpg".format(k0))
    shutil.move(src,dst)
    k0+=1

unfilter_fnames = os.listdir(train_plastic_dir)
filter_fnames = sample(unfilter_fnames,4)
for fname in filter_fnames:
    src = os.path.join(train_plastic_dir,fname)
    dst = os.path.join(testdir,"plastic{}.jpg".format(m0))
    shutil.move(src,dst)
    m0+=1

unfilter_fnames = os.listdir(train_glass_dir)
filter_fnames = sample(unfilter_fnames,4)
for fname in filter_fnames:
    src = os.path.join(train_glass_dir,fname)
    dst = os.path.join(testdir,"glass{}.jpg".format(i0))
    shutil.move(src,dst)
    i0+=1
# unfilter_fnames = os.listdir(train_glass_dir)
# for i in unfilter_fnames:
#     i = train_glass_dir+'\\'+i
#     os.rename(i,r"{}\\glass{}.jpg".format(train_glass_dir,i1))
#     i1+=1
#
# unfilter_fnames = os.listdir(train_plastic_dir)
# for i in unfilter_fnames:
#     i = train_plastic_dir+'\\'+i
#     os.rename(i,r"{}\\plastic{}.jpg".format(train_glass_dir,j1))
#     j1+=1
#
# unfilter_fnames = os.listdir(train_paper_dir)
# for i in unfilter_fnames:
#     i = train_paper_dir+'\\'+i
#     os.rename(i,r"{}\\paper{}.jpg".format(train_glass_dir,k1))
#     k1+=1
#
# unfilter_fnames = os.listdir(train_metal_dir)
# for i in unfilter_fnames:
#     i = train_metal_dir+'\\'+i
#     os.rename(i,r"{}\\metal{}.jpg".format(train_glass_dir,m1))
#     m1+=1