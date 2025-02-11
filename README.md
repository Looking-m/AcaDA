# AcaDA
# Prerequisites:
python == 3.9<br>
pytorch == 1.1.0<br>
torchvision == 0.3.0<br>
numpy, scipy, sklearn, tqdm, argparse

# Datasets
Please manually download the datasets [Office](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view), [Office-Home](https://accounts.google.com/InteractiveLogin/signinchooser?continue=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F0B81rNlvomiwed0V1YUxQdC1uOTg%2Fview&followup=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F0B81rNlvomiwed0V1YUxQdC1uOTg%2Fview&osid=1&passive=1209600&service=wise&ifkv=ASSHykrILX_LKBTofh-_9qs_vFUKYvH7fwG6eBzkxagN08D1M6iRvLZSE1A4SkOMQ-S2EzjFURZ_rA&ddm=1&flowName=GlifWebSignIn&flowEntry=ServiceLogin), DomainNet from the official websites

# Training
'''
python image_source.py --trte val --da uda --output ckps/source/ --gpu_id 0 --dset office --s 0  

'''

'''
python image_target.py --cls_par 0.3 --da uda --output_src ckps/source/ --output ckps/target/ --dset office --s 0  

'''


