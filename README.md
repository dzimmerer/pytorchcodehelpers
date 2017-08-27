# Pytorch Code Helpers

Just some small helpers while coding pytorch. 

They basically allow a very rough inspection into your current models/ code. 

INSTALL (as pycharm external tool):
---------

1. Download repo with ```git clone```.
2. Install repo with ```pip install -e pytorchcodehelpers```
3. Open pycharm, go to _Settings -> Tools -> External Tools_ and two new tools:

    (a) Name: `(e. g.) PytorchCode`, 
        Programm: `$JDKPath$` ,
        Parameters: `-m pytorchcodehelpers.pytcodetool $FilePathRelativeToProjectRoot$ $FilePath$ $SelectionStartLine$ $SelectionEndLine$` ,
        Working directory: `$ProjectFileDir$`
        
     (b) Name: `(e. g.) PytorchModel`, 
        Programm: `$JDKPath$` ,
        Parameters: `-m pytorchcodehelpers.pytmodeltool  $FilePathRelativeToProjectRoot$  $SelectedText$` ,
        Working directory: `$ProjectFileDir$`
         
4. Done =) , perhaps restart pycharm, and test the tool (with right click)


USAGE
------

**PytorchCode:**

Simply mark the code you want to evaluate in pycharm e. g. :
```python

    torch.nn.Conv2d(3, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    torch.nn.LeakyReLU(0.2)
    torch.nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
    torch.nn.LeakyReLU(0.2)
    torch.nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
    torch.nn.LeakyReLU(0.2)
    torch.nn.Conv2d(512, 1024, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True)
    torch.nn.LeakyReLU(0.2)
    torch.nn.Conv2d(1024, 256, kernel_size=[8, 8], stride=(1, 1), bias=False)
```
 Click right, go to External Tools and choose PytorchCode. 
 Now if every thing works well, you should be asked for the input size, so enter it and that's it (now if it works, it should tell you the dimensions, and possibly generate you a class/ pytorch module)
 
 You can also (try to) add tensor opertions like torch.cat() and name variables (for later reuse, "out of the boy it assumes a linear tensor flow ;-P')

_**Right now this is very hacky code, so be careful !!!**_ 
It does not yet support muliple lines in the function definiton, multiline doc strings, any controll flow operations (for, if, else, ...) and class definitions.


**PytorchModel:**

Simply mark the model e.g. 
```python

torchvision.models.AlexNet()

```

 Click right, go to External Tools and choose PytorchModel. 
 If everything worked well it should display you the dimensions of the model.
 
 
 Expected Outputs:
 ----------------
 
 **PytorchCode:** 
 
 ```
Give the input size (default: 1 3 128 128) : 1 3 128 128
inpt        :  (1, 3, 128, 128)
//
conv2d0     :  (1, 128, 64, 64)
leakyrelu0  :  (1, 128, 64, 64)
conv2d1     :  (1, 256, 32, 32)
batchnorm2d0  :  (1, 256, 32, 32)
leakyrelu1  :  (1, 256, 32, 32)
conv2d2     :  (1, 512, 16, 16)
batchnorm2d1  :  (1, 512, 16, 16)
leakyrelu2  :  (1, 512, 16, 16)
conv2d3     :  (1, 1024, 8, 8)
batchnorm2d2  :  (1, 1024, 8, 8)
leakyrelu3  :  (1, 1024, 8, 8)
conv2d4     :  (1, 256, 1, 1)
//
//
Get automatic generated class ? (default: N) : Y
Class name ? (default: NewModule) : 
//
//
class NewModule(torch.nn.Module):

	def __init__(self, ):
		super(NewModule, self).__init__()
		self.m_conv2d0 = torch.nn.Conv2d(3, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
		self.m_leakyrelu0 = torch.nn.LeakyReLU(0.2)
		self.m_conv2d1 = torch.nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
		self.m_batchnorm2d0 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
		self.m_leakyrelu1 = torch.nn.LeakyReLU(0.2)
		self.m_conv2d2 = torch.nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
		self.m_batchnorm2d1 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
		self.m_leakyrelu2 = torch.nn.LeakyReLU(0.2)
		self.m_conv2d3 = torch.nn.Conv2d(512, 1024, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
		self.m_batchnorm2d2 = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True)
		self.m_leakyrelu3 = torch.nn.LeakyReLU(0.2)
		self.m_conv2d4 = torch.nn.Conv2d(1024, 256, kernel_size=[8, 8], stride=(1, 1), bias=False)
		
		

	def forward(self, inpt, ):
		conv2d0 = self.m_conv2d0(inpt)
		leakyrelu0 = self.m_leakyrelu0(conv2d0)
		conv2d1 = self.m_conv2d1(leakyrelu0)
		batchnorm2d0 = self.m_batchnorm2d0(conv2d1)
		leakyrelu1 = self.m_leakyrelu1(batchnorm2d0)
		conv2d2 = self.m_conv2d2(leakyrelu1)
		batchnorm2d1 = self.m_batchnorm2d1(conv2d2)
		leakyrelu2 = self.m_leakyrelu2(batchnorm2d1)
		conv2d3 = self.m_conv2d3(leakyrelu2)
		batchnorm2d2 = self.m_batchnorm2d2(conv2d3)
		leakyrelu3 = self.m_leakyrelu3(batchnorm2d2)
		conv2d4 = self.m_conv2d4(leakyrelu3)
		
		
		return 
//
//
Done.

```
 
 
 **PytorchModel:**
 
 ```
Give the input size (eg. 1 3 128 128, if non given will do nothing) : 1 3 224 224
 (in)  AlexNet         : (1, 3, 224, 224)
   (in)  Sequential      : (1, 3, 224, 224)
     (in)  Conv2d          : (1, 3, 224, 224)
     (out) Conv2d          : (64, 55, 55)
     (in)  ReLU            : (1, 64, 55, 55)
     (out) ReLU            : (64, 55, 55)
     (in)  MaxPool2d       : (1, 64, 55, 55)
     (out) MaxPool2d       : (64, 27, 27)
     (in)  Conv2d          : (1, 64, 27, 27)
     (out) Conv2d          : (192, 27, 27)
     (in)  ReLU            : (1, 192, 27, 27)
     (out) ReLU            : (192, 27, 27)
     (in)  MaxPool2d       : (1, 192, 27, 27)
     (out) MaxPool2d       : (192, 13, 13)
     (in)  Conv2d          : (1, 192, 13, 13)
     (out) Conv2d          : (384, 13, 13)
     (in)  ReLU            : (1, 384, 13, 13)
     (out) ReLU            : (384, 13, 13)
     (in)  Conv2d          : (1, 384, 13, 13)
     (out) Conv2d          : (256, 13, 13)
     (in)  ReLU            : (1, 256, 13, 13)
     (out) ReLU            : (256, 13, 13)
     (in)  Conv2d          : (1, 256, 13, 13)
     (out) Conv2d          : (256, 13, 13)
     (in)  ReLU            : (1, 256, 13, 13)
     (out) ReLU            : (256, 13, 13)
     (in)  MaxPool2d       : (1, 256, 13, 13)
     (out) MaxPool2d       : (256, 6, 6)
   (out) Sequential      : (256, 6, 6)
   (in)  Sequential      : (1, 9216)
     (in)  Dropout         : (1, 9216)
     (out) Dropout         : (9216,)
     (in)  Linear          : (1, 9216)
     (out) Linear          : (4096,)
     (in)  ReLU            : (1, 4096)
     (out) ReLU            : (4096,)
     (in)  Dropout         : (1, 4096)
     (out) Dropout         : (4096,)
     (in)  Linear          : (1, 4096)
     (out) Linear          : (4096,)
     (in)  ReLU            : (1, 4096)
     (out) ReLU            : (4096,)
     (in)  Linear          : (1, 4096)
     (out) Linear          : (1000,)
   (out) Sequential      : (1000,)
 (out) AlexNet         : (1000,)

Done.


```
 
 DISCLAIMER
-----------

This is very bad code and very hacky, so use on your own risks.
I'm grateful for feedback / bugreports but can not promise to work on those, since it is primarily a convience feature for me.




