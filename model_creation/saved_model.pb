??+
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.1.02unknown8??%
?
conv2d_1_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameconv2d_1_1/kernel

%conv2d_1_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1_1/kernel*&
_output_shapes
:@*
dtype0
v
conv2d_1_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_1_1/bias
o
#conv2d_1_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1_1/bias*
_output_shapes
:@*
dtype0
~
batchnorm_1_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_namebatchnorm_1_1/gamma
w
'batchnorm_1_1/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_1_1/gamma*
_output_shapes
:@*
dtype0
|
batchnorm_1_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_namebatchnorm_1_1/beta
u
&batchnorm_1_1/beta/Read/ReadVariableOpReadVariableOpbatchnorm_1_1/beta*
_output_shapes
:@*
dtype0
?
batchnorm_1_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namebatchnorm_1_1/moving_mean
?
-batchnorm_1_1/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_1_1/moving_mean*
_output_shapes
:@*
dtype0
?
batchnorm_1_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_namebatchnorm_1_1/moving_variance
?
1batchnorm_1_1/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_1_1/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_2_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*"
shared_nameconv2d_2_1/kernel

%conv2d_2_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_2_1/kernel*&
_output_shapes
:@@*
dtype0
v
conv2d_2_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_2_1/bias
o
#conv2d_2_1/bias/Read/ReadVariableOpReadVariableOpconv2d_2_1/bias*
_output_shapes
:@*
dtype0
~
batchnorm_2_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_namebatchnorm_2_1/gamma
w
'batchnorm_2_1/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_2_1/gamma*
_output_shapes
:@*
dtype0
|
batchnorm_2_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_namebatchnorm_2_1/beta
u
&batchnorm_2_1/beta/Read/ReadVariableOpReadVariableOpbatchnorm_2_1/beta*
_output_shapes
:@*
dtype0
?
batchnorm_2_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namebatchnorm_2_1/moving_mean
?
-batchnorm_2_1/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_2_1/moving_mean*
_output_shapes
:@*
dtype0
?
batchnorm_2_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_namebatchnorm_2_1/moving_variance
?
1batchnorm_2_1/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_2_1/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_3_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*"
shared_nameconv2d_3_1/kernel
?
%conv2d_3_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_3_1/kernel*'
_output_shapes
:@?*
dtype0
w
conv2d_3_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?* 
shared_nameconv2d_3_1/bias
p
#conv2d_3_1/bias/Read/ReadVariableOpReadVariableOpconv2d_3_1/bias*
_output_shapes	
:?*
dtype0

batchnorm_3_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_namebatchnorm_3_1/gamma
x
'batchnorm_3_1/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_3_1/gamma*
_output_shapes	
:?*
dtype0
}
batchnorm_3_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*#
shared_namebatchnorm_3_1/beta
v
&batchnorm_3_1/beta/Read/ReadVariableOpReadVariableOpbatchnorm_3_1/beta*
_output_shapes	
:?*
dtype0
?
batchnorm_3_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namebatchnorm_3_1/moving_mean
?
-batchnorm_3_1/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_3_1/moving_mean*
_output_shapes	
:?*
dtype0
?
batchnorm_3_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*.
shared_namebatchnorm_3_1/moving_variance
?
1batchnorm_3_1/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_3_1/moving_variance*
_output_shapes	
:?*
dtype0
?
conv2d_4_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*"
shared_nameconv2d_4_1/kernel
?
%conv2d_4_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_4_1/kernel*(
_output_shapes
:??*
dtype0
w
conv2d_4_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?* 
shared_nameconv2d_4_1/bias
p
#conv2d_4_1/bias/Read/ReadVariableOpReadVariableOpconv2d_4_1/bias*
_output_shapes	
:?*
dtype0

batchnorm_4_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_namebatchnorm_4_1/gamma
x
'batchnorm_4_1/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_4_1/gamma*
_output_shapes	
:?*
dtype0
}
batchnorm_4_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*#
shared_namebatchnorm_4_1/beta
v
&batchnorm_4_1/beta/Read/ReadVariableOpReadVariableOpbatchnorm_4_1/beta*
_output_shapes	
:?*
dtype0
?
batchnorm_4_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namebatchnorm_4_1/moving_mean
?
-batchnorm_4_1/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_4_1/moving_mean*
_output_shapes	
:?*
dtype0
?
batchnorm_4_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*.
shared_namebatchnorm_4_1/moving_variance
?
1batchnorm_4_1/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_4_1/moving_variance*
_output_shapes	
:?*
dtype0
?
conv2d_5_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*"
shared_nameconv2d_5_1/kernel
?
%conv2d_5_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_5_1/kernel*(
_output_shapes
:??*
dtype0
w
conv2d_5_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?* 
shared_nameconv2d_5_1/bias
p
#conv2d_5_1/bias/Read/ReadVariableOpReadVariableOpconv2d_5_1/bias*
_output_shapes	
:?*
dtype0

batchnorm_5_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_namebatchnorm_5_1/gamma
x
'batchnorm_5_1/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_5_1/gamma*
_output_shapes	
:?*
dtype0
}
batchnorm_5_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*#
shared_namebatchnorm_5_1/beta
v
&batchnorm_5_1/beta/Read/ReadVariableOpReadVariableOpbatchnorm_5_1/beta*
_output_shapes	
:?*
dtype0
?
batchnorm_5_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namebatchnorm_5_1/moving_mean
?
-batchnorm_5_1/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_5_1/moving_mean*
_output_shapes	
:?*
dtype0
?
batchnorm_5_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*.
shared_namebatchnorm_5_1/moving_variance
?
1batchnorm_5_1/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_5_1/moving_variance*
_output_shapes	
:?*
dtype0
?
conv2d_6_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*"
shared_nameconv2d_6_1/kernel
?
%conv2d_6_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_6_1/kernel*(
_output_shapes
:??*
dtype0
w
conv2d_6_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?* 
shared_nameconv2d_6_1/bias
p
#conv2d_6_1/bias/Read/ReadVariableOpReadVariableOpconv2d_6_1/bias*
_output_shapes	
:?*
dtype0

batchnorm_6_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_namebatchnorm_6_1/gamma
x
'batchnorm_6_1/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_6_1/gamma*
_output_shapes	
:?*
dtype0
}
batchnorm_6_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*#
shared_namebatchnorm_6_1/beta
v
&batchnorm_6_1/beta/Read/ReadVariableOpReadVariableOpbatchnorm_6_1/beta*
_output_shapes	
:?*
dtype0
?
batchnorm_6_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namebatchnorm_6_1/moving_mean
?
-batchnorm_6_1/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_6_1/moving_mean*
_output_shapes	
:?*
dtype0
?
batchnorm_6_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*.
shared_namebatchnorm_6_1/moving_variance
?
1batchnorm_6_1/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_6_1/moving_variance*
_output_shapes	
:?*
dtype0
~
dense_1_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?H?*!
shared_namedense_1_1/kernel
w
$dense_1_1/kernel/Read/ReadVariableOpReadVariableOpdense_1_1/kernel* 
_output_shapes
:
?H?*
dtype0
u
dense_1_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_1_1/bias
n
"dense_1_1/bias/Read/ReadVariableOpReadVariableOpdense_1_1/bias*
_output_shapes	
:?*
dtype0

batchnorm_7_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_namebatchnorm_7_1/gamma
x
'batchnorm_7_1/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_7_1/gamma*
_output_shapes	
:?*
dtype0
}
batchnorm_7_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*#
shared_namebatchnorm_7_1/beta
v
&batchnorm_7_1/beta/Read/ReadVariableOpReadVariableOpbatchnorm_7_1/beta*
_output_shapes	
:?*
dtype0
?
batchnorm_7_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namebatchnorm_7_1/moving_mean
?
-batchnorm_7_1/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_7_1/moving_mean*
_output_shapes	
:?*
dtype0
?
batchnorm_7_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*.
shared_namebatchnorm_7_1/moving_variance
?
1batchnorm_7_1/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_7_1/moving_variance*
_output_shapes	
:?*
dtype0
?
out_layer_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*#
shared_nameout_layer_1/kernel
z
&out_layer_1/kernel/Read/ReadVariableOpReadVariableOpout_layer_1/kernel*
_output_shapes
:	?*
dtype0
x
out_layer_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameout_layer_1/bias
q
$out_layer_1/bias/Read/ReadVariableOpReadVariableOpout_layer_1/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
Adam/conv2d_1_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/conv2d_1_1/kernel/m
?
,Adam/conv2d_1_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1_1/kernel/m*&
_output_shapes
:@*
dtype0
?
Adam/conv2d_1_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_1_1/bias/m
}
*Adam/conv2d_1_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1_1/bias/m*
_output_shapes
:@*
dtype0
?
Adam/batchnorm_1_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameAdam/batchnorm_1_1/gamma/m
?
.Adam/batchnorm_1_1/gamma/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_1_1/gamma/m*
_output_shapes
:@*
dtype0
?
Adam/batchnorm_1_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameAdam/batchnorm_1_1/beta/m
?
-Adam/batchnorm_1_1/beta/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_1_1/beta/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_2_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*)
shared_nameAdam/conv2d_2_1/kernel/m
?
,Adam/conv2d_2_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2_1/kernel/m*&
_output_shapes
:@@*
dtype0
?
Adam/conv2d_2_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_2_1/bias/m
}
*Adam/conv2d_2_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2_1/bias/m*
_output_shapes
:@*
dtype0
?
Adam/batchnorm_2_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameAdam/batchnorm_2_1/gamma/m
?
.Adam/batchnorm_2_1/gamma/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_2_1/gamma/m*
_output_shapes
:@*
dtype0
?
Adam/batchnorm_2_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameAdam/batchnorm_2_1/beta/m
?
-Adam/batchnorm_2_1/beta/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_2_1/beta/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_3_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*)
shared_nameAdam/conv2d_3_1/kernel/m
?
,Adam/conv2d_3_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3_1/kernel/m*'
_output_shapes
:@?*
dtype0
?
Adam/conv2d_3_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_nameAdam/conv2d_3_1/bias/m
~
*Adam/conv2d_3_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3_1/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/batchnorm_3_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nameAdam/batchnorm_3_1/gamma/m
?
.Adam/batchnorm_3_1/gamma/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_3_1/gamma/m*
_output_shapes	
:?*
dtype0
?
Adam/batchnorm_3_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_nameAdam/batchnorm_3_1/beta/m
?
-Adam/batchnorm_3_1/beta/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_3_1/beta/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_4_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*)
shared_nameAdam/conv2d_4_1/kernel/m
?
,Adam/conv2d_4_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4_1/kernel/m*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_4_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_nameAdam/conv2d_4_1/bias/m
~
*Adam/conv2d_4_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4_1/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/batchnorm_4_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nameAdam/batchnorm_4_1/gamma/m
?
.Adam/batchnorm_4_1/gamma/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_4_1/gamma/m*
_output_shapes	
:?*
dtype0
?
Adam/batchnorm_4_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_nameAdam/batchnorm_4_1/beta/m
?
-Adam/batchnorm_4_1/beta/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_4_1/beta/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_5_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*)
shared_nameAdam/conv2d_5_1/kernel/m
?
,Adam/conv2d_5_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5_1/kernel/m*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_5_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_nameAdam/conv2d_5_1/bias/m
~
*Adam/conv2d_5_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5_1/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/batchnorm_5_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nameAdam/batchnorm_5_1/gamma/m
?
.Adam/batchnorm_5_1/gamma/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_5_1/gamma/m*
_output_shapes	
:?*
dtype0
?
Adam/batchnorm_5_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_nameAdam/batchnorm_5_1/beta/m
?
-Adam/batchnorm_5_1/beta/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_5_1/beta/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_6_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*)
shared_nameAdam/conv2d_6_1/kernel/m
?
,Adam/conv2d_6_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6_1/kernel/m*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_6_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_nameAdam/conv2d_6_1/bias/m
~
*Adam/conv2d_6_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6_1/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/batchnorm_6_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nameAdam/batchnorm_6_1/gamma/m
?
.Adam/batchnorm_6_1/gamma/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_6_1/gamma/m*
_output_shapes	
:?*
dtype0
?
Adam/batchnorm_6_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_nameAdam/batchnorm_6_1/beta/m
?
-Adam/batchnorm_6_1/beta/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_6_1/beta/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_1_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?H?*(
shared_nameAdam/dense_1_1/kernel/m
?
+Adam/dense_1_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1_1/kernel/m* 
_output_shapes
:
?H?*
dtype0
?
Adam/dense_1_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/dense_1_1/bias/m
|
)Adam/dense_1_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1_1/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/batchnorm_7_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nameAdam/batchnorm_7_1/gamma/m
?
.Adam/batchnorm_7_1/gamma/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_7_1/gamma/m*
_output_shapes	
:?*
dtype0
?
Adam/batchnorm_7_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_nameAdam/batchnorm_7_1/beta/m
?
-Adam/batchnorm_7_1/beta/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_7_1/beta/m*
_output_shapes	
:?*
dtype0
?
Adam/out_layer_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?**
shared_nameAdam/out_layer_1/kernel/m
?
-Adam/out_layer_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/out_layer_1/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/out_layer_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/out_layer_1/bias/m

+Adam/out_layer_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/out_layer_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_1_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/conv2d_1_1/kernel/v
?
,Adam/conv2d_1_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1_1/kernel/v*&
_output_shapes
:@*
dtype0
?
Adam/conv2d_1_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_1_1/bias/v
}
*Adam/conv2d_1_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1_1/bias/v*
_output_shapes
:@*
dtype0
?
Adam/batchnorm_1_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameAdam/batchnorm_1_1/gamma/v
?
.Adam/batchnorm_1_1/gamma/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_1_1/gamma/v*
_output_shapes
:@*
dtype0
?
Adam/batchnorm_1_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameAdam/batchnorm_1_1/beta/v
?
-Adam/batchnorm_1_1/beta/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_1_1/beta/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_2_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*)
shared_nameAdam/conv2d_2_1/kernel/v
?
,Adam/conv2d_2_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2_1/kernel/v*&
_output_shapes
:@@*
dtype0
?
Adam/conv2d_2_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_2_1/bias/v
}
*Adam/conv2d_2_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2_1/bias/v*
_output_shapes
:@*
dtype0
?
Adam/batchnorm_2_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameAdam/batchnorm_2_1/gamma/v
?
.Adam/batchnorm_2_1/gamma/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_2_1/gamma/v*
_output_shapes
:@*
dtype0
?
Adam/batchnorm_2_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameAdam/batchnorm_2_1/beta/v
?
-Adam/batchnorm_2_1/beta/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_2_1/beta/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_3_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*)
shared_nameAdam/conv2d_3_1/kernel/v
?
,Adam/conv2d_3_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3_1/kernel/v*'
_output_shapes
:@?*
dtype0
?
Adam/conv2d_3_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_nameAdam/conv2d_3_1/bias/v
~
*Adam/conv2d_3_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3_1/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/batchnorm_3_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nameAdam/batchnorm_3_1/gamma/v
?
.Adam/batchnorm_3_1/gamma/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_3_1/gamma/v*
_output_shapes	
:?*
dtype0
?
Adam/batchnorm_3_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_nameAdam/batchnorm_3_1/beta/v
?
-Adam/batchnorm_3_1/beta/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_3_1/beta/v*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_4_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*)
shared_nameAdam/conv2d_4_1/kernel/v
?
,Adam/conv2d_4_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4_1/kernel/v*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_4_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_nameAdam/conv2d_4_1/bias/v
~
*Adam/conv2d_4_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4_1/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/batchnorm_4_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nameAdam/batchnorm_4_1/gamma/v
?
.Adam/batchnorm_4_1/gamma/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_4_1/gamma/v*
_output_shapes	
:?*
dtype0
?
Adam/batchnorm_4_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_nameAdam/batchnorm_4_1/beta/v
?
-Adam/batchnorm_4_1/beta/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_4_1/beta/v*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_5_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*)
shared_nameAdam/conv2d_5_1/kernel/v
?
,Adam/conv2d_5_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5_1/kernel/v*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_5_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_nameAdam/conv2d_5_1/bias/v
~
*Adam/conv2d_5_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5_1/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/batchnorm_5_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nameAdam/batchnorm_5_1/gamma/v
?
.Adam/batchnorm_5_1/gamma/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_5_1/gamma/v*
_output_shapes	
:?*
dtype0
?
Adam/batchnorm_5_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_nameAdam/batchnorm_5_1/beta/v
?
-Adam/batchnorm_5_1/beta/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_5_1/beta/v*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_6_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*)
shared_nameAdam/conv2d_6_1/kernel/v
?
,Adam/conv2d_6_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6_1/kernel/v*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_6_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_nameAdam/conv2d_6_1/bias/v
~
*Adam/conv2d_6_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6_1/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/batchnorm_6_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nameAdam/batchnorm_6_1/gamma/v
?
.Adam/batchnorm_6_1/gamma/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_6_1/gamma/v*
_output_shapes	
:?*
dtype0
?
Adam/batchnorm_6_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_nameAdam/batchnorm_6_1/beta/v
?
-Adam/batchnorm_6_1/beta/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_6_1/beta/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_1_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?H?*(
shared_nameAdam/dense_1_1/kernel/v
?
+Adam/dense_1_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1_1/kernel/v* 
_output_shapes
:
?H?*
dtype0
?
Adam/dense_1_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/dense_1_1/bias/v
|
)Adam/dense_1_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1_1/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/batchnorm_7_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nameAdam/batchnorm_7_1/gamma/v
?
.Adam/batchnorm_7_1/gamma/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_7_1/gamma/v*
_output_shapes	
:?*
dtype0
?
Adam/batchnorm_7_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_nameAdam/batchnorm_7_1/beta/v
?
-Adam/batchnorm_7_1/beta/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_7_1/beta/v*
_output_shapes	
:?*
dtype0
?
Adam/out_layer_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?**
shared_nameAdam/out_layer_1/kernel/v
?
-Adam/out_layer_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/out_layer_1/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/out_layer_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/out_layer_1/bias/v

+Adam/out_layer_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/out_layer_1/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
Դ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer-11
layer-12
layer_with_weights-8
layer-13
layer_with_weights-9
layer-14
layer_with_weights-10
layer-15
layer_with_weights-11
layer-16
layer-17
layer-18
layer-19
layer_with_weights-12
layer-20
layer_with_weights-13
layer-21
layer-22
layer_with_weights-14
layer-23
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
h

kernel
 bias
!regularization_losses
"trainable_variables
#	variables
$	keras_api
?
%axis
	&gamma
'beta
(moving_mean
)moving_variance
*regularization_losses
+trainable_variables
,	variables
-	keras_api
h

.kernel
/bias
0regularization_losses
1trainable_variables
2	variables
3	keras_api
?
4axis
	5gamma
6beta
7moving_mean
8moving_variance
9regularization_losses
:trainable_variables
;	variables
<	keras_api
R
=regularization_losses
>trainable_variables
?	variables
@	keras_api
R
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
h

Ekernel
Fbias
Gregularization_losses
Htrainable_variables
I	variables
J	keras_api
?
Kaxis
	Lgamma
Mbeta
Nmoving_mean
Omoving_variance
Pregularization_losses
Qtrainable_variables
R	variables
S	keras_api
h

Tkernel
Ubias
Vregularization_losses
Wtrainable_variables
X	variables
Y	keras_api
?
Zaxis
	[gamma
\beta
]moving_mean
^moving_variance
_regularization_losses
`trainable_variables
a	variables
b	keras_api
R
cregularization_losses
dtrainable_variables
e	variables
f	keras_api
R
gregularization_losses
htrainable_variables
i	variables
j	keras_api
h

kkernel
lbias
mregularization_losses
ntrainable_variables
o	variables
p	keras_api
?
qaxis
	rgamma
sbeta
tmoving_mean
umoving_variance
vregularization_losses
wtrainable_variables
x	variables
y	keras_api
h

zkernel
{bias
|regularization_losses
}trainable_variables
~	variables
	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?iter
?beta_1
?beta_2

?decay
?learning_ratem? m?&m?'m?.m?/m?5m?6m?Em?Fm?Lm?Mm?Tm?Um?[m?\m?km?lm?rm?sm?zm?{m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?v? v?&v?'v?.v?/v?5v?6v?Ev?Fv?Lv?Mv?Tv?Uv?[v?\v?kv?lv?rv?sv?zv?{v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?
 
?
0
 1
&2
'3
.4
/5
56
67
E8
F9
L10
M11
T12
U13
[14
\15
k16
l17
r18
s19
z20
{21
?22
?23
?24
?25
?26
?27
?28
?29
?
0
 1
&2
'3
(4
)5
.6
/7
58
69
710
811
E12
F13
L14
M15
N16
O17
T18
U19
[20
\21
]22
^23
k24
l25
r26
s27
t28
u29
z30
{31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?
?layers
regularization_losses
?metrics
trainable_variables
	variables
 ?layer_regularization_losses
?non_trainable_variables
 
][
VARIABLE_VALUEconv2d_1_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_1_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
 1

0
 1
?
?layers
!regularization_losses
?metrics
"trainable_variables
#	variables
 ?layer_regularization_losses
?non_trainable_variables
 
^\
VARIABLE_VALUEbatchnorm_1_1/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEbatchnorm_1_1/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEbatchnorm_1_1/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEbatchnorm_1_1/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

&0
'1

&0
'1
(2
)3
?
?layers
*regularization_losses
?metrics
+trainable_variables
,	variables
 ?layer_regularization_losses
?non_trainable_variables
][
VARIABLE_VALUEconv2d_2_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_2_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

.0
/1

.0
/1
?
?layers
0regularization_losses
?metrics
1trainable_variables
2	variables
 ?layer_regularization_losses
?non_trainable_variables
 
^\
VARIABLE_VALUEbatchnorm_2_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEbatchnorm_2_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEbatchnorm_2_1/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEbatchnorm_2_1/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

50
61

50
61
72
83
?
?layers
9regularization_losses
?metrics
:trainable_variables
;	variables
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
?layers
=regularization_losses
?metrics
>trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
?layers
Aregularization_losses
?metrics
Btrainable_variables
C	variables
 ?layer_regularization_losses
?non_trainable_variables
][
VARIABLE_VALUEconv2d_3_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_3_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

E0
F1

E0
F1
?
?layers
Gregularization_losses
?metrics
Htrainable_variables
I	variables
 ?layer_regularization_losses
?non_trainable_variables
 
^\
VARIABLE_VALUEbatchnorm_3_1/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEbatchnorm_3_1/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEbatchnorm_3_1/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEbatchnorm_3_1/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

L0
M1

L0
M1
N2
O3
?
?layers
Pregularization_losses
?metrics
Qtrainable_variables
R	variables
 ?layer_regularization_losses
?non_trainable_variables
][
VARIABLE_VALUEconv2d_4_1/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_4_1/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

T0
U1

T0
U1
?
?layers
Vregularization_losses
?metrics
Wtrainable_variables
X	variables
 ?layer_regularization_losses
?non_trainable_variables
 
^\
VARIABLE_VALUEbatchnorm_4_1/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEbatchnorm_4_1/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEbatchnorm_4_1/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEbatchnorm_4_1/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

[0
\1

[0
\1
]2
^3
?
?layers
_regularization_losses
?metrics
`trainable_variables
a	variables
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
?layers
cregularization_losses
?metrics
dtrainable_variables
e	variables
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
?layers
gregularization_losses
?metrics
htrainable_variables
i	variables
 ?layer_regularization_losses
?non_trainable_variables
][
VARIABLE_VALUEconv2d_5_1/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_5_1/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

k0
l1

k0
l1
?
?layers
mregularization_losses
?metrics
ntrainable_variables
o	variables
 ?layer_regularization_losses
?non_trainable_variables
 
^\
VARIABLE_VALUEbatchnorm_5_1/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEbatchnorm_5_1/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEbatchnorm_5_1/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEbatchnorm_5_1/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

r0
s1

r0
s1
t2
u3
?
?layers
vregularization_losses
?metrics
wtrainable_variables
x	variables
 ?layer_regularization_losses
?non_trainable_variables
^\
VARIABLE_VALUEconv2d_6_1/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv2d_6_1/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 

z0
{1

z0
{1
?
?layers
|regularization_losses
?metrics
}trainable_variables
~	variables
 ?layer_regularization_losses
?non_trainable_variables
 
_]
VARIABLE_VALUEbatchnorm_6_1/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEbatchnorm_6_1/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEbatchnorm_6_1/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEbatchnorm_6_1/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1
 
?0
?1
?2
?3
?
?layers
?regularization_losses
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
?layers
?regularization_losses
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
?layers
?regularization_losses
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
?layers
?regularization_losses
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
][
VARIABLE_VALUEdense_1_1/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_1_1/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?layers
?regularization_losses
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
 
_]
VARIABLE_VALUEbatchnorm_7_1/gamma6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEbatchnorm_7_1/beta5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEbatchnorm_7_1/moving_mean<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEbatchnorm_7_1/moving_variance@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1
 
?0
?1
?2
?3
?
?layers
?regularization_losses
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
?layers
?regularization_losses
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
_]
VARIABLE_VALUEout_layer_1/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEout_layer_1/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?layers
?regularization_losses
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
?
0
1
2
3
4
5
6
	7

8
9
10
11
12
13
14
15
16
17
18
19
20
21
22

?0
 
j
(0
)1
72
83
N4
O5
]6
^7
t8
u9
?10
?11
?12
?13
 
 
 
 
 
 
 

(0
)1
 
 
 
 
 
 
 

70
81
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

N0
O1
 
 
 
 
 
 
 

]0
^1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

t0
u1
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 


?total

?count
?
_fn_kwargs
?regularization_losses
?trainable_variables
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

?0
?1
?
?layers
?regularization_losses
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
 
 
 

?0
?1
?~
VARIABLE_VALUEAdam/conv2d_1_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_1_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/batchnorm_1_1/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/batchnorm_1_1/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_2_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_2_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/batchnorm_2_1/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/batchnorm_2_1/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_3_1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_3_1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/batchnorm_3_1/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/batchnorm_3_1/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_4_1/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_4_1/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/batchnorm_4_1/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/batchnorm_4_1/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_5_1/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_5_1/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/batchnorm_5_1/gamma/mQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/batchnorm_5_1/beta/mPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/conv2d_6_1/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv2d_6_1/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/batchnorm_6_1/gamma/mRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/batchnorm_6_1/beta/mQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/dense_1_1/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_1_1/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/batchnorm_7_1/gamma/mRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/batchnorm_7_1/beta/mQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/out_layer_1/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/out_layer_1/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_1_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_1_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/batchnorm_1_1/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/batchnorm_1_1/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_2_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_2_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/batchnorm_2_1/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/batchnorm_2_1/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_3_1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_3_1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/batchnorm_3_1/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/batchnorm_3_1/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_4_1/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_4_1/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/batchnorm_4_1/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/batchnorm_4_1/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_5_1/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_5_1/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/batchnorm_5_1/gamma/vQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/batchnorm_5_1/beta/vPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/conv2d_6_1/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv2d_6_1/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/batchnorm_6_1/gamma/vRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/batchnorm_6_1/beta/vQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/dense_1_1/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_1_1/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/batchnorm_7_1/gamma/vRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/batchnorm_7_1/beta/vQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/out_layer_1/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/out_layer_1/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_conv2d_1_inputPlaceholder*/
_output_shapes
:?????????00*
dtype0*$
shape:?????????00
?

StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_1_inputconv2d_1_1/kernelconv2d_1_1/biasbatchnorm_1_1/gammabatchnorm_1_1/betabatchnorm_1_1/moving_meanbatchnorm_1_1/moving_varianceconv2d_2_1/kernelconv2d_2_1/biasbatchnorm_2_1/gammabatchnorm_2_1/betabatchnorm_2_1/moving_meanbatchnorm_2_1/moving_varianceconv2d_3_1/kernelconv2d_3_1/biasbatchnorm_3_1/gammabatchnorm_3_1/betabatchnorm_3_1/moving_meanbatchnorm_3_1/moving_varianceconv2d_4_1/kernelconv2d_4_1/biasbatchnorm_4_1/gammabatchnorm_4_1/betabatchnorm_4_1/moving_meanbatchnorm_4_1/moving_varianceconv2d_5_1/kernelconv2d_5_1/biasbatchnorm_5_1/gammabatchnorm_5_1/betabatchnorm_5_1/moving_meanbatchnorm_5_1/moving_varianceconv2d_6_1/kernelconv2d_6_1/biasbatchnorm_6_1/gammabatchnorm_6_1/betabatchnorm_6_1/moving_meanbatchnorm_6_1/moving_variancedense_1_1/kerneldense_1_1/biasbatchnorm_7_1/moving_variancebatchnorm_7_1/gammabatchnorm_7_1/moving_meanbatchnorm_7_1/betaout_layer_1/kernelout_layer_1/bias*8
Tin1
/2-*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*,
f'R%
#__inference_signature_wrapper_27384
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?(
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%conv2d_1_1/kernel/Read/ReadVariableOp#conv2d_1_1/bias/Read/ReadVariableOp'batchnorm_1_1/gamma/Read/ReadVariableOp&batchnorm_1_1/beta/Read/ReadVariableOp-batchnorm_1_1/moving_mean/Read/ReadVariableOp1batchnorm_1_1/moving_variance/Read/ReadVariableOp%conv2d_2_1/kernel/Read/ReadVariableOp#conv2d_2_1/bias/Read/ReadVariableOp'batchnorm_2_1/gamma/Read/ReadVariableOp&batchnorm_2_1/beta/Read/ReadVariableOp-batchnorm_2_1/moving_mean/Read/ReadVariableOp1batchnorm_2_1/moving_variance/Read/ReadVariableOp%conv2d_3_1/kernel/Read/ReadVariableOp#conv2d_3_1/bias/Read/ReadVariableOp'batchnorm_3_1/gamma/Read/ReadVariableOp&batchnorm_3_1/beta/Read/ReadVariableOp-batchnorm_3_1/moving_mean/Read/ReadVariableOp1batchnorm_3_1/moving_variance/Read/ReadVariableOp%conv2d_4_1/kernel/Read/ReadVariableOp#conv2d_4_1/bias/Read/ReadVariableOp'batchnorm_4_1/gamma/Read/ReadVariableOp&batchnorm_4_1/beta/Read/ReadVariableOp-batchnorm_4_1/moving_mean/Read/ReadVariableOp1batchnorm_4_1/moving_variance/Read/ReadVariableOp%conv2d_5_1/kernel/Read/ReadVariableOp#conv2d_5_1/bias/Read/ReadVariableOp'batchnorm_5_1/gamma/Read/ReadVariableOp&batchnorm_5_1/beta/Read/ReadVariableOp-batchnorm_5_1/moving_mean/Read/ReadVariableOp1batchnorm_5_1/moving_variance/Read/ReadVariableOp%conv2d_6_1/kernel/Read/ReadVariableOp#conv2d_6_1/bias/Read/ReadVariableOp'batchnorm_6_1/gamma/Read/ReadVariableOp&batchnorm_6_1/beta/Read/ReadVariableOp-batchnorm_6_1/moving_mean/Read/ReadVariableOp1batchnorm_6_1/moving_variance/Read/ReadVariableOp$dense_1_1/kernel/Read/ReadVariableOp"dense_1_1/bias/Read/ReadVariableOp'batchnorm_7_1/gamma/Read/ReadVariableOp&batchnorm_7_1/beta/Read/ReadVariableOp-batchnorm_7_1/moving_mean/Read/ReadVariableOp1batchnorm_7_1/moving_variance/Read/ReadVariableOp&out_layer_1/kernel/Read/ReadVariableOp$out_layer_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp,Adam/conv2d_1_1/kernel/m/Read/ReadVariableOp*Adam/conv2d_1_1/bias/m/Read/ReadVariableOp.Adam/batchnorm_1_1/gamma/m/Read/ReadVariableOp-Adam/batchnorm_1_1/beta/m/Read/ReadVariableOp,Adam/conv2d_2_1/kernel/m/Read/ReadVariableOp*Adam/conv2d_2_1/bias/m/Read/ReadVariableOp.Adam/batchnorm_2_1/gamma/m/Read/ReadVariableOp-Adam/batchnorm_2_1/beta/m/Read/ReadVariableOp,Adam/conv2d_3_1/kernel/m/Read/ReadVariableOp*Adam/conv2d_3_1/bias/m/Read/ReadVariableOp.Adam/batchnorm_3_1/gamma/m/Read/ReadVariableOp-Adam/batchnorm_3_1/beta/m/Read/ReadVariableOp,Adam/conv2d_4_1/kernel/m/Read/ReadVariableOp*Adam/conv2d_4_1/bias/m/Read/ReadVariableOp.Adam/batchnorm_4_1/gamma/m/Read/ReadVariableOp-Adam/batchnorm_4_1/beta/m/Read/ReadVariableOp,Adam/conv2d_5_1/kernel/m/Read/ReadVariableOp*Adam/conv2d_5_1/bias/m/Read/ReadVariableOp.Adam/batchnorm_5_1/gamma/m/Read/ReadVariableOp-Adam/batchnorm_5_1/beta/m/Read/ReadVariableOp,Adam/conv2d_6_1/kernel/m/Read/ReadVariableOp*Adam/conv2d_6_1/bias/m/Read/ReadVariableOp.Adam/batchnorm_6_1/gamma/m/Read/ReadVariableOp-Adam/batchnorm_6_1/beta/m/Read/ReadVariableOp+Adam/dense_1_1/kernel/m/Read/ReadVariableOp)Adam/dense_1_1/bias/m/Read/ReadVariableOp.Adam/batchnorm_7_1/gamma/m/Read/ReadVariableOp-Adam/batchnorm_7_1/beta/m/Read/ReadVariableOp-Adam/out_layer_1/kernel/m/Read/ReadVariableOp+Adam/out_layer_1/bias/m/Read/ReadVariableOp,Adam/conv2d_1_1/kernel/v/Read/ReadVariableOp*Adam/conv2d_1_1/bias/v/Read/ReadVariableOp.Adam/batchnorm_1_1/gamma/v/Read/ReadVariableOp-Adam/batchnorm_1_1/beta/v/Read/ReadVariableOp,Adam/conv2d_2_1/kernel/v/Read/ReadVariableOp*Adam/conv2d_2_1/bias/v/Read/ReadVariableOp.Adam/batchnorm_2_1/gamma/v/Read/ReadVariableOp-Adam/batchnorm_2_1/beta/v/Read/ReadVariableOp,Adam/conv2d_3_1/kernel/v/Read/ReadVariableOp*Adam/conv2d_3_1/bias/v/Read/ReadVariableOp.Adam/batchnorm_3_1/gamma/v/Read/ReadVariableOp-Adam/batchnorm_3_1/beta/v/Read/ReadVariableOp,Adam/conv2d_4_1/kernel/v/Read/ReadVariableOp*Adam/conv2d_4_1/bias/v/Read/ReadVariableOp.Adam/batchnorm_4_1/gamma/v/Read/ReadVariableOp-Adam/batchnorm_4_1/beta/v/Read/ReadVariableOp,Adam/conv2d_5_1/kernel/v/Read/ReadVariableOp*Adam/conv2d_5_1/bias/v/Read/ReadVariableOp.Adam/batchnorm_5_1/gamma/v/Read/ReadVariableOp-Adam/batchnorm_5_1/beta/v/Read/ReadVariableOp,Adam/conv2d_6_1/kernel/v/Read/ReadVariableOp*Adam/conv2d_6_1/bias/v/Read/ReadVariableOp.Adam/batchnorm_6_1/gamma/v/Read/ReadVariableOp-Adam/batchnorm_6_1/beta/v/Read/ReadVariableOp+Adam/dense_1_1/kernel/v/Read/ReadVariableOp)Adam/dense_1_1/bias/v/Read/ReadVariableOp.Adam/batchnorm_7_1/gamma/v/Read/ReadVariableOp-Adam/batchnorm_7_1/beta/v/Read/ReadVariableOp-Adam/out_layer_1/kernel/v/Read/ReadVariableOp+Adam/out_layer_1/bias/v/Read/ReadVariableOpConst*|
Tinu
s2q	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

CPU

GPU 2J 8*'
f"R 
__inference__traced_save_29642
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_1_1/kernelconv2d_1_1/biasbatchnorm_1_1/gammabatchnorm_1_1/betabatchnorm_1_1/moving_meanbatchnorm_1_1/moving_varianceconv2d_2_1/kernelconv2d_2_1/biasbatchnorm_2_1/gammabatchnorm_2_1/betabatchnorm_2_1/moving_meanbatchnorm_2_1/moving_varianceconv2d_3_1/kernelconv2d_3_1/biasbatchnorm_3_1/gammabatchnorm_3_1/betabatchnorm_3_1/moving_meanbatchnorm_3_1/moving_varianceconv2d_4_1/kernelconv2d_4_1/biasbatchnorm_4_1/gammabatchnorm_4_1/betabatchnorm_4_1/moving_meanbatchnorm_4_1/moving_varianceconv2d_5_1/kernelconv2d_5_1/biasbatchnorm_5_1/gammabatchnorm_5_1/betabatchnorm_5_1/moving_meanbatchnorm_5_1/moving_varianceconv2d_6_1/kernelconv2d_6_1/biasbatchnorm_6_1/gammabatchnorm_6_1/betabatchnorm_6_1/moving_meanbatchnorm_6_1/moving_variancedense_1_1/kerneldense_1_1/biasbatchnorm_7_1/gammabatchnorm_7_1/betabatchnorm_7_1/moving_meanbatchnorm_7_1/moving_varianceout_layer_1/kernelout_layer_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv2d_1_1/kernel/mAdam/conv2d_1_1/bias/mAdam/batchnorm_1_1/gamma/mAdam/batchnorm_1_1/beta/mAdam/conv2d_2_1/kernel/mAdam/conv2d_2_1/bias/mAdam/batchnorm_2_1/gamma/mAdam/batchnorm_2_1/beta/mAdam/conv2d_3_1/kernel/mAdam/conv2d_3_1/bias/mAdam/batchnorm_3_1/gamma/mAdam/batchnorm_3_1/beta/mAdam/conv2d_4_1/kernel/mAdam/conv2d_4_1/bias/mAdam/batchnorm_4_1/gamma/mAdam/batchnorm_4_1/beta/mAdam/conv2d_5_1/kernel/mAdam/conv2d_5_1/bias/mAdam/batchnorm_5_1/gamma/mAdam/batchnorm_5_1/beta/mAdam/conv2d_6_1/kernel/mAdam/conv2d_6_1/bias/mAdam/batchnorm_6_1/gamma/mAdam/batchnorm_6_1/beta/mAdam/dense_1_1/kernel/mAdam/dense_1_1/bias/mAdam/batchnorm_7_1/gamma/mAdam/batchnorm_7_1/beta/mAdam/out_layer_1/kernel/mAdam/out_layer_1/bias/mAdam/conv2d_1_1/kernel/vAdam/conv2d_1_1/bias/vAdam/batchnorm_1_1/gamma/vAdam/batchnorm_1_1/beta/vAdam/conv2d_2_1/kernel/vAdam/conv2d_2_1/bias/vAdam/batchnorm_2_1/gamma/vAdam/batchnorm_2_1/beta/vAdam/conv2d_3_1/kernel/vAdam/conv2d_3_1/bias/vAdam/batchnorm_3_1/gamma/vAdam/batchnorm_3_1/beta/vAdam/conv2d_4_1/kernel/vAdam/conv2d_4_1/bias/vAdam/batchnorm_4_1/gamma/vAdam/batchnorm_4_1/beta/vAdam/conv2d_5_1/kernel/vAdam/conv2d_5_1/bias/vAdam/batchnorm_5_1/gamma/vAdam/batchnorm_5_1/beta/vAdam/conv2d_6_1/kernel/vAdam/conv2d_6_1/bias/vAdam/batchnorm_6_1/gamma/vAdam/batchnorm_6_1/beta/vAdam/dense_1_1/kernel/vAdam/dense_1_1/bias/vAdam/batchnorm_7_1/gamma/vAdam/batchnorm_7_1/beta/vAdam/out_layer_1/kernel/vAdam/out_layer_1/bias/v*{
Tint
r2p*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

CPU

GPU 2J 8**
f%R#
!__inference__traced_restore_29987׉"
?
?
+__inference_batchnorm_4_layer_call_fn_28697

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_4_layer_call_and_return_conditional_losses_265182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
+__inference_batchnorm_6_layer_call_fn_29052

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_6_layer_call_and_return_conditional_losses_267212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
+__inference_batchnorm_2_layer_call_fn_28268

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_2_layer_call_and_return_conditional_losses_253722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
+__inference_batchnorm_2_layer_call_fn_28259

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_2_layer_call_and_return_conditional_losses_253412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
F__inference_batchnorm_5_layer_call_and_return_conditional_losses_28800

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
Const?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?
?
F__inference_batchnorm_4_layer_call_and_return_conditional_losses_25690

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
Const?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?#
?
F__inference_batchnorm_5_layer_call_and_return_conditional_losses_28778

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_28763
assignmovingavg_1_28770
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *?p}?2	
Const_2?
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/28763*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/28763*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_28763*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/28763*
_output_shapes	
:?2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/28763*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_28763AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/28763*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/28770*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/28770*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_28770*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/28770*
_output_shapes	
:?2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/28770*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_28770AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/28770*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?
?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_25086

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAddo
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Elu?
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
?
b
F__inference_maxpool2d_3_layer_call_and_return_conditional_losses_26021

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?
?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_25557

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAddp
EluEluBiasAdd:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
Elu?
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
F__inference_batchnorm_7_layer_call_and_return_conditional_losses_29214

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs
?#
?
F__inference_batchnorm_6_layer_call_and_return_conditional_losses_26699

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_26684
assignmovingavg_1_26691
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *?p}?2	
Const_2?
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/26684*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/26684*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_26684*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/26684*
_output_shapes	
:?2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/26684*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_26684AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/26684*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/26691*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/26691*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_26691*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/26691*
_output_shapes	
:?2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/26691*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_26691AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/26691*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?
?
+__inference_batchnorm_5_layer_call_fn_28809

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_5_layer_call_and_return_conditional_losses_266172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
??
?
?__inference_DCNN_layer_call_and_return_conditional_losses_27924

inputs+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource'
#batchnorm_1_readvariableop_resource)
%batchnorm_1_readvariableop_1_resource8
4batchnorm_1_fusedbatchnormv3_readvariableop_resource:
6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource'
#batchnorm_2_readvariableop_resource)
%batchnorm_2_readvariableop_1_resource8
4batchnorm_2_fusedbatchnormv3_readvariableop_resource:
6batchnorm_2_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource'
#batchnorm_3_readvariableop_resource)
%batchnorm_3_readvariableop_1_resource8
4batchnorm_3_fusedbatchnormv3_readvariableop_resource:
6batchnorm_3_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource'
#batchnorm_4_readvariableop_resource)
%batchnorm_4_readvariableop_1_resource8
4batchnorm_4_fusedbatchnormv3_readvariableop_resource:
6batchnorm_4_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource'
#batchnorm_5_readvariableop_resource)
%batchnorm_5_readvariableop_1_resource8
4batchnorm_5_fusedbatchnormv3_readvariableop_resource:
6batchnorm_5_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource'
#batchnorm_6_readvariableop_resource)
%batchnorm_6_readvariableop_1_resource8
4batchnorm_6_fusedbatchnormv3_readvariableop_resource:
6batchnorm_6_fusedbatchnormv3_readvariableop_1_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource1
-batchnorm_7_batchnorm_readvariableop_resource5
1batchnorm_7_batchnorm_mul_readvariableop_resource3
/batchnorm_7_batchnorm_readvariableop_1_resource3
/batchnorm_7_batchnorm_readvariableop_2_resource,
(out_layer_matmul_readvariableop_resource-
)out_layer_biasadd_readvariableop_resource
identity??+batchnorm_1/FusedBatchNormV3/ReadVariableOp?-batchnorm_1/FusedBatchNormV3/ReadVariableOp_1?batchnorm_1/ReadVariableOp?batchnorm_1/ReadVariableOp_1?+batchnorm_2/FusedBatchNormV3/ReadVariableOp?-batchnorm_2/FusedBatchNormV3/ReadVariableOp_1?batchnorm_2/ReadVariableOp?batchnorm_2/ReadVariableOp_1?+batchnorm_3/FusedBatchNormV3/ReadVariableOp?-batchnorm_3/FusedBatchNormV3/ReadVariableOp_1?batchnorm_3/ReadVariableOp?batchnorm_3/ReadVariableOp_1?+batchnorm_4/FusedBatchNormV3/ReadVariableOp?-batchnorm_4/FusedBatchNormV3/ReadVariableOp_1?batchnorm_4/ReadVariableOp?batchnorm_4/ReadVariableOp_1?+batchnorm_5/FusedBatchNormV3/ReadVariableOp?-batchnorm_5/FusedBatchNormV3/ReadVariableOp_1?batchnorm_5/ReadVariableOp?batchnorm_5/ReadVariableOp_1?+batchnorm_6/FusedBatchNormV3/ReadVariableOp?-batchnorm_6/FusedBatchNormV3/ReadVariableOp_1?batchnorm_6/ReadVariableOp?batchnorm_6/ReadVariableOp_1?$batchnorm_7/batchnorm/ReadVariableOp?&batchnorm_7/batchnorm/ReadVariableOp_1?&batchnorm_7/batchnorm/ReadVariableOp_2?(batchnorm_7/batchnorm/mul/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp? out_layer/BiasAdd/ReadVariableOp?out_layer/MatMul/ReadVariableOp?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00@*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00@2
conv2d_1/BiasAddx
conv2d_1/EluEluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00@2
conv2d_1/Eluv
batchnorm_1/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
batchnorm_1/LogicalAnd/xv
batchnorm_1/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
batchnorm_1/LogicalAnd/y?
batchnorm_1/LogicalAnd
LogicalAnd!batchnorm_1/LogicalAnd/x:output:0!batchnorm_1/LogicalAnd/y:output:0*
_output_shapes
: 2
batchnorm_1/LogicalAnd?
batchnorm_1/ReadVariableOpReadVariableOp#batchnorm_1_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm_1/ReadVariableOp?
batchnorm_1/ReadVariableOp_1ReadVariableOp%batchnorm_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm_1/ReadVariableOp_1?
+batchnorm_1/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02-
+batchnorm_1/FusedBatchNormV3/ReadVariableOp?
-batchnorm_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02/
-batchnorm_1/FusedBatchNormV3/ReadVariableOp_1?
batchnorm_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/Elu:activations:0"batchnorm_1/ReadVariableOp:value:0$batchnorm_1/ReadVariableOp_1:value:03batchnorm_1/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00@:@:@:@:@:*
epsilon%o?:*
is_training( 2
batchnorm_1/FusedBatchNormV3k
batchnorm_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
batchnorm_1/Const?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2D batchnorm_1/FusedBatchNormV3:y:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00@*
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00@2
conv2d_2/BiasAddx
conv2d_2/EluEluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00@2
conv2d_2/Eluv
batchnorm_2/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
batchnorm_2/LogicalAnd/xv
batchnorm_2/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
batchnorm_2/LogicalAnd/y?
batchnorm_2/LogicalAnd
LogicalAnd!batchnorm_2/LogicalAnd/x:output:0!batchnorm_2/LogicalAnd/y:output:0*
_output_shapes
: 2
batchnorm_2/LogicalAnd?
batchnorm_2/ReadVariableOpReadVariableOp#batchnorm_2_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm_2/ReadVariableOp?
batchnorm_2/ReadVariableOp_1ReadVariableOp%batchnorm_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm_2/ReadVariableOp_1?
+batchnorm_2/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02-
+batchnorm_2/FusedBatchNormV3/ReadVariableOp?
-batchnorm_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02/
-batchnorm_2/FusedBatchNormV3/ReadVariableOp_1?
batchnorm_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/Elu:activations:0"batchnorm_2/ReadVariableOp:value:0$batchnorm_2/ReadVariableOp_1:value:03batchnorm_2/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00@:@:@:@:@:*
epsilon%o?:*
is_training( 2
batchnorm_2/FusedBatchNormV3k
batchnorm_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
batchnorm_2/Const?
maxpool2d_1/MaxPoolMaxPool batchnorm_2/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
maxpool2d_1/MaxPool?
dropout_1/IdentityIdentitymaxpool2d_1/MaxPool:output:0*
T0*/
_output_shapes
:?????????@2
dropout_1/Identity?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Ddropout_1/Identity:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_3/BiasAddy
conv2d_3/EluEluconv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_3/Eluv
batchnorm_3/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
batchnorm_3/LogicalAnd/xv
batchnorm_3/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
batchnorm_3/LogicalAnd/y?
batchnorm_3/LogicalAnd
LogicalAnd!batchnorm_3/LogicalAnd/x:output:0!batchnorm_3/LogicalAnd/y:output:0*
_output_shapes
: 2
batchnorm_3/LogicalAnd?
batchnorm_3/ReadVariableOpReadVariableOp#batchnorm_3_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm_3/ReadVariableOp?
batchnorm_3/ReadVariableOp_1ReadVariableOp%batchnorm_3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm_3/ReadVariableOp_1?
+batchnorm_3/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+batchnorm_3/FusedBatchNormV3/ReadVariableOp?
-batchnorm_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02/
-batchnorm_3/FusedBatchNormV3/ReadVariableOp_1?
batchnorm_3/FusedBatchNormV3FusedBatchNormV3conv2d_3/Elu:activations:0"batchnorm_3/ReadVariableOp:value:0$batchnorm_3/ReadVariableOp_1:value:03batchnorm_3/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
batchnorm_3/FusedBatchNormV3k
batchnorm_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
batchnorm_3/Const?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2D batchnorm_3/FusedBatchNormV3:y:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_4/BiasAddy
conv2d_4/EluEluconv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_4/Eluv
batchnorm_4/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
batchnorm_4/LogicalAnd/xv
batchnorm_4/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
batchnorm_4/LogicalAnd/y?
batchnorm_4/LogicalAnd
LogicalAnd!batchnorm_4/LogicalAnd/x:output:0!batchnorm_4/LogicalAnd/y:output:0*
_output_shapes
: 2
batchnorm_4/LogicalAnd?
batchnorm_4/ReadVariableOpReadVariableOp#batchnorm_4_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm_4/ReadVariableOp?
batchnorm_4/ReadVariableOp_1ReadVariableOp%batchnorm_4_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm_4/ReadVariableOp_1?
+batchnorm_4/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+batchnorm_4/FusedBatchNormV3/ReadVariableOp?
-batchnorm_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02/
-batchnorm_4/FusedBatchNormV3/ReadVariableOp_1?
batchnorm_4/FusedBatchNormV3FusedBatchNormV3conv2d_4/Elu:activations:0"batchnorm_4/ReadVariableOp:value:0$batchnorm_4/ReadVariableOp_1:value:03batchnorm_4/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
batchnorm_4/FusedBatchNormV3k
batchnorm_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
batchnorm_4/Const?
maxpool2d_2/MaxPoolMaxPool batchnorm_4/FusedBatchNormV3:y:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
maxpool2d_2/MaxPool?
dropout_2/IdentityIdentitymaxpool2d_2/MaxPool:output:0*
T0*0
_output_shapes
:??????????2
dropout_2/Identity?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2Ddropout_2/Identity:output:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_5/BiasAddy
conv2d_5/EluEluconv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_5/Eluv
batchnorm_5/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
batchnorm_5/LogicalAnd/xv
batchnorm_5/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
batchnorm_5/LogicalAnd/y?
batchnorm_5/LogicalAnd
LogicalAnd!batchnorm_5/LogicalAnd/x:output:0!batchnorm_5/LogicalAnd/y:output:0*
_output_shapes
: 2
batchnorm_5/LogicalAnd?
batchnorm_5/ReadVariableOpReadVariableOp#batchnorm_5_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm_5/ReadVariableOp?
batchnorm_5/ReadVariableOp_1ReadVariableOp%batchnorm_5_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm_5/ReadVariableOp_1?
+batchnorm_5/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+batchnorm_5/FusedBatchNormV3/ReadVariableOp?
-batchnorm_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02/
-batchnorm_5/FusedBatchNormV3/ReadVariableOp_1?
batchnorm_5/FusedBatchNormV3FusedBatchNormV3conv2d_5/Elu:activations:0"batchnorm_5/ReadVariableOp:value:0$batchnorm_5/ReadVariableOp_1:value:03batchnorm_5/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
batchnorm_5/FusedBatchNormV3k
batchnorm_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
batchnorm_5/Const?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2D batchnorm_5/FusedBatchNormV3:y:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_6/BiasAddy
conv2d_6/EluEluconv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_6/Eluv
batchnorm_6/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
batchnorm_6/LogicalAnd/xv
batchnorm_6/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
batchnorm_6/LogicalAnd/y?
batchnorm_6/LogicalAnd
LogicalAnd!batchnorm_6/LogicalAnd/x:output:0!batchnorm_6/LogicalAnd/y:output:0*
_output_shapes
: 2
batchnorm_6/LogicalAnd?
batchnorm_6/ReadVariableOpReadVariableOp#batchnorm_6_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm_6/ReadVariableOp?
batchnorm_6/ReadVariableOp_1ReadVariableOp%batchnorm_6_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm_6/ReadVariableOp_1?
+batchnorm_6/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+batchnorm_6/FusedBatchNormV3/ReadVariableOp?
-batchnorm_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02/
-batchnorm_6/FusedBatchNormV3/ReadVariableOp_1?
batchnorm_6/FusedBatchNormV3FusedBatchNormV3conv2d_6/Elu:activations:0"batchnorm_6/ReadVariableOp:value:0$batchnorm_6/ReadVariableOp_1:value:03batchnorm_6/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
batchnorm_6/FusedBatchNormV3k
batchnorm_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
batchnorm_6/Const?
maxpool2d_3/MaxPoolMaxPool batchnorm_6/FusedBatchNormV3:y:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
maxpool2d_3/MaxPool?
dropout_3/IdentityIdentitymaxpool2d_3/MaxPool:output:0*
T0*0
_output_shapes
:??????????2
dropout_3/Identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? $  2
flatten/Const?
flatten/ReshapeReshapedropout_3/Identity:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????H2
flatten/Reshape?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
?H?*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddn
dense_1/EluEludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_1/Eluv
batchnorm_7/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
batchnorm_7/LogicalAnd/xv
batchnorm_7/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
batchnorm_7/LogicalAnd/y?
batchnorm_7/LogicalAnd
LogicalAnd!batchnorm_7/LogicalAnd/x:output:0!batchnorm_7/LogicalAnd/y:output:0*
_output_shapes
: 2
batchnorm_7/LogicalAnd?
$batchnorm_7/batchnorm/ReadVariableOpReadVariableOp-batchnorm_7_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batchnorm_7/batchnorm/ReadVariableOp
batchnorm_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm_7/batchnorm/add/y?
batchnorm_7/batchnorm/addAddV2,batchnorm_7/batchnorm/ReadVariableOp:value:0$batchnorm_7/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm_7/batchnorm/add?
batchnorm_7/batchnorm/RsqrtRsqrtbatchnorm_7/batchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm_7/batchnorm/Rsqrt?
(batchnorm_7/batchnorm/mul/ReadVariableOpReadVariableOp1batchnorm_7_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(batchnorm_7/batchnorm/mul/ReadVariableOp?
batchnorm_7/batchnorm/mulMulbatchnorm_7/batchnorm/Rsqrt:y:00batchnorm_7/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm_7/batchnorm/mul?
batchnorm_7/batchnorm/mul_1Muldense_1/Elu:activations:0batchnorm_7/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm_7/batchnorm/mul_1?
&batchnorm_7/batchnorm/ReadVariableOp_1ReadVariableOp/batchnorm_7_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batchnorm_7/batchnorm/ReadVariableOp_1?
batchnorm_7/batchnorm/mul_2Mul.batchnorm_7/batchnorm/ReadVariableOp_1:value:0batchnorm_7/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm_7/batchnorm/mul_2?
&batchnorm_7/batchnorm/ReadVariableOp_2ReadVariableOp/batchnorm_7_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02(
&batchnorm_7/batchnorm/ReadVariableOp_2?
batchnorm_7/batchnorm/subSub.batchnorm_7/batchnorm/ReadVariableOp_2:value:0batchnorm_7/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm_7/batchnorm/sub?
batchnorm_7/batchnorm/add_1AddV2batchnorm_7/batchnorm/mul_1:z:0batchnorm_7/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm_7/batchnorm/add_1?
dropout_4/IdentityIdentitybatchnorm_7/batchnorm/add_1:z:0*
T0*(
_output_shapes
:??????????2
dropout_4/Identity?
out_layer/MatMul/ReadVariableOpReadVariableOp(out_layer_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02!
out_layer/MatMul/ReadVariableOp?
out_layer/MatMulMatMuldropout_4/Identity:output:0'out_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
out_layer/MatMul?
 out_layer/BiasAdd/ReadVariableOpReadVariableOp)out_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 out_layer/BiasAdd/ReadVariableOp?
out_layer/BiasAddBiasAddout_layer/MatMul:product:0(out_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
out_layer/BiasAdd
out_layer/SoftmaxSoftmaxout_layer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
out_layer/Softmax?
IdentityIdentityout_layer/Softmax:softmax:0,^batchnorm_1/FusedBatchNormV3/ReadVariableOp.^batchnorm_1/FusedBatchNormV3/ReadVariableOp_1^batchnorm_1/ReadVariableOp^batchnorm_1/ReadVariableOp_1,^batchnorm_2/FusedBatchNormV3/ReadVariableOp.^batchnorm_2/FusedBatchNormV3/ReadVariableOp_1^batchnorm_2/ReadVariableOp^batchnorm_2/ReadVariableOp_1,^batchnorm_3/FusedBatchNormV3/ReadVariableOp.^batchnorm_3/FusedBatchNormV3/ReadVariableOp_1^batchnorm_3/ReadVariableOp^batchnorm_3/ReadVariableOp_1,^batchnorm_4/FusedBatchNormV3/ReadVariableOp.^batchnorm_4/FusedBatchNormV3/ReadVariableOp_1^batchnorm_4/ReadVariableOp^batchnorm_4/ReadVariableOp_1,^batchnorm_5/FusedBatchNormV3/ReadVariableOp.^batchnorm_5/FusedBatchNormV3/ReadVariableOp_1^batchnorm_5/ReadVariableOp^batchnorm_5/ReadVariableOp_1,^batchnorm_6/FusedBatchNormV3/ReadVariableOp.^batchnorm_6/FusedBatchNormV3/ReadVariableOp_1^batchnorm_6/ReadVariableOp^batchnorm_6/ReadVariableOp_1%^batchnorm_7/batchnorm/ReadVariableOp'^batchnorm_7/batchnorm/ReadVariableOp_1'^batchnorm_7/batchnorm/ReadVariableOp_2)^batchnorm_7/batchnorm/mul/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp!^out_layer/BiasAdd/ReadVariableOp ^out_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????00::::::::::::::::::::::::::::::::::::::::::::2Z
+batchnorm_1/FusedBatchNormV3/ReadVariableOp+batchnorm_1/FusedBatchNormV3/ReadVariableOp2^
-batchnorm_1/FusedBatchNormV3/ReadVariableOp_1-batchnorm_1/FusedBatchNormV3/ReadVariableOp_128
batchnorm_1/ReadVariableOpbatchnorm_1/ReadVariableOp2<
batchnorm_1/ReadVariableOp_1batchnorm_1/ReadVariableOp_12Z
+batchnorm_2/FusedBatchNormV3/ReadVariableOp+batchnorm_2/FusedBatchNormV3/ReadVariableOp2^
-batchnorm_2/FusedBatchNormV3/ReadVariableOp_1-batchnorm_2/FusedBatchNormV3/ReadVariableOp_128
batchnorm_2/ReadVariableOpbatchnorm_2/ReadVariableOp2<
batchnorm_2/ReadVariableOp_1batchnorm_2/ReadVariableOp_12Z
+batchnorm_3/FusedBatchNormV3/ReadVariableOp+batchnorm_3/FusedBatchNormV3/ReadVariableOp2^
-batchnorm_3/FusedBatchNormV3/ReadVariableOp_1-batchnorm_3/FusedBatchNormV3/ReadVariableOp_128
batchnorm_3/ReadVariableOpbatchnorm_3/ReadVariableOp2<
batchnorm_3/ReadVariableOp_1batchnorm_3/ReadVariableOp_12Z
+batchnorm_4/FusedBatchNormV3/ReadVariableOp+batchnorm_4/FusedBatchNormV3/ReadVariableOp2^
-batchnorm_4/FusedBatchNormV3/ReadVariableOp_1-batchnorm_4/FusedBatchNormV3/ReadVariableOp_128
batchnorm_4/ReadVariableOpbatchnorm_4/ReadVariableOp2<
batchnorm_4/ReadVariableOp_1batchnorm_4/ReadVariableOp_12Z
+batchnorm_5/FusedBatchNormV3/ReadVariableOp+batchnorm_5/FusedBatchNormV3/ReadVariableOp2^
-batchnorm_5/FusedBatchNormV3/ReadVariableOp_1-batchnorm_5/FusedBatchNormV3/ReadVariableOp_128
batchnorm_5/ReadVariableOpbatchnorm_5/ReadVariableOp2<
batchnorm_5/ReadVariableOp_1batchnorm_5/ReadVariableOp_12Z
+batchnorm_6/FusedBatchNormV3/ReadVariableOp+batchnorm_6/FusedBatchNormV3/ReadVariableOp2^
-batchnorm_6/FusedBatchNormV3/ReadVariableOp_1-batchnorm_6/FusedBatchNormV3/ReadVariableOp_128
batchnorm_6/ReadVariableOpbatchnorm_6/ReadVariableOp2<
batchnorm_6/ReadVariableOp_1batchnorm_6/ReadVariableOp_12L
$batchnorm_7/batchnorm/ReadVariableOp$batchnorm_7/batchnorm/ReadVariableOp2P
&batchnorm_7/batchnorm/ReadVariableOp_1&batchnorm_7/batchnorm/ReadVariableOp_12P
&batchnorm_7/batchnorm/ReadVariableOp_2&batchnorm_7/batchnorm/ReadVariableOp_22T
(batchnorm_7/batchnorm/mul/ReadVariableOp(batchnorm_7/batchnorm/mul/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2D
 out_layer/BiasAdd/ReadVariableOp out_layer/BiasAdd/ReadVariableOp2B
out_layer/MatMul/ReadVariableOpout_layer/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?#
?
F__inference_batchnorm_2_layer_call_and_return_conditional_losses_28302

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_28287
assignmovingavg_1_28294
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:?????????00@:@:@:@:@:*
epsilon%o?:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *?p}?2	
Const_2?
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/28287*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/28287*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_28287*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/28287*
_output_shapes
:@2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/28287*
_output_shapes
:@2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_28287AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/28287*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/28294*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/28294*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_28294*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/28294*
_output_shapes
:@2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/28294*
_output_shapes
:@2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_28294AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/28294*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????00@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????00@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?
c
D__inference_dropout_3_layer_call_and_return_conditional_losses_26766

inputs
identity?a
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/random_uniform/max?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniform?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*0
_output_shapes
:??????????2
dropout/random_uniform/mul?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*0
_output_shapes
:??????????2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*0
_output_shapes
:??????????2
dropout/GreaterEqualy
dropout/mulMulinputsdropout/truediv:z:0*
T0*0
_output_shapes
:??????????2
dropout/mul?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
dropout/Cast?
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
dropout/mul_1n
IdentityIdentitydropout/mul_1:z:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
?
F__inference_batchnorm_2_layer_call_and_return_conditional_losses_28250

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
Const?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?
?
+__inference_batchnorm_1_layer_call_fn_28173

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????00@**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_1_layer_call_and_return_conditional_losses_262112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????00@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????00@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_28362

inputs
identity?a
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/random_uniform/max?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniform?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:?????????@2
dropout/random_uniform/mul?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:?????????@2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*/
_output_shapes
:?????????@2
dropout/GreaterEqualx
dropout/mulMulinputsdropout/truediv:z:0*
T0*/
_output_shapes
:?????????@2
dropout/mul?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
dropout/Cast?
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
dropout/mul_1m
IdentityIdentitydropout/mul_1:z:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:& "
 
_user_specified_nameinputs
?
?
+__inference_batchnorm_5_layer_call_fn_28892

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_5_layer_call_and_return_conditional_losses_258552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
F__inference_batchnorm_5_layer_call_and_return_conditional_losses_25855

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
Const?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?/
?
F__inference_batchnorm_7_layer_call_and_return_conditional_losses_26132

inputs
assignmovingavg_26107
assignmovingavg_1_26113)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst*(
_class
loc:@AssignMovingAvg/26107*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_26107*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*(
_class
loc:@AssignMovingAvg/26107*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*(
_class
loc:@AssignMovingAvg/26107*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_26107AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/26107*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst**
_class 
loc:@AssignMovingAvg_1/26113*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_26113*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/26113*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/26113*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_26113AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/26113*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
?__inference_DCNN_layer_call_and_return_conditional_losses_26906
conv2d_1_input+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2.
*batchnorm_1_statefulpartitionedcall_args_1.
*batchnorm_1_statefulpartitionedcall_args_2.
*batchnorm_1_statefulpartitionedcall_args_3.
*batchnorm_1_statefulpartitionedcall_args_4+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2.
*batchnorm_2_statefulpartitionedcall_args_1.
*batchnorm_2_statefulpartitionedcall_args_2.
*batchnorm_2_statefulpartitionedcall_args_3.
*batchnorm_2_statefulpartitionedcall_args_4+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_2.
*batchnorm_3_statefulpartitionedcall_args_1.
*batchnorm_3_statefulpartitionedcall_args_2.
*batchnorm_3_statefulpartitionedcall_args_3.
*batchnorm_3_statefulpartitionedcall_args_4+
'conv2d_4_statefulpartitionedcall_args_1+
'conv2d_4_statefulpartitionedcall_args_2.
*batchnorm_4_statefulpartitionedcall_args_1.
*batchnorm_4_statefulpartitionedcall_args_2.
*batchnorm_4_statefulpartitionedcall_args_3.
*batchnorm_4_statefulpartitionedcall_args_4+
'conv2d_5_statefulpartitionedcall_args_1+
'conv2d_5_statefulpartitionedcall_args_2.
*batchnorm_5_statefulpartitionedcall_args_1.
*batchnorm_5_statefulpartitionedcall_args_2.
*batchnorm_5_statefulpartitionedcall_args_3.
*batchnorm_5_statefulpartitionedcall_args_4+
'conv2d_6_statefulpartitionedcall_args_1+
'conv2d_6_statefulpartitionedcall_args_2.
*batchnorm_6_statefulpartitionedcall_args_1.
*batchnorm_6_statefulpartitionedcall_args_2.
*batchnorm_6_statefulpartitionedcall_args_3.
*batchnorm_6_statefulpartitionedcall_args_4*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2.
*batchnorm_7_statefulpartitionedcall_args_1.
*batchnorm_7_statefulpartitionedcall_args_2.
*batchnorm_7_statefulpartitionedcall_args_3.
*batchnorm_7_statefulpartitionedcall_args_4,
(out_layer_statefulpartitionedcall_args_1,
(out_layer_statefulpartitionedcall_args_2
identity??#batchnorm_1/StatefulPartitionedCall?#batchnorm_2/StatefulPartitionedCall?#batchnorm_3/StatefulPartitionedCall?#batchnorm_4/StatefulPartitionedCall?#batchnorm_5/StatefulPartitionedCall?#batchnorm_6/StatefulPartitionedCall?#batchnorm_7/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?!dropout_3/StatefulPartitionedCall?!dropout_4/StatefulPartitionedCall?!out_layer/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallconv2d_1_input'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????00@**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_250862"
 conv2d_1/StatefulPartitionedCall?
#batchnorm_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*batchnorm_1_statefulpartitionedcall_args_1*batchnorm_1_statefulpartitionedcall_args_2*batchnorm_1_statefulpartitionedcall_args_3*batchnorm_1_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????00@**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_1_layer_call_and_return_conditional_losses_262112%
#batchnorm_1/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall,batchnorm_1/StatefulPartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????00@**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_252392"
 conv2d_2/StatefulPartitionedCall?
#batchnorm_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*batchnorm_2_statefulpartitionedcall_args_1*batchnorm_2_statefulpartitionedcall_args_2*batchnorm_2_statefulpartitionedcall_args_3*batchnorm_2_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????00@**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_2_layer_call_and_return_conditional_losses_262932%
#batchnorm_2/StatefulPartitionedCall?
maxpool2d_1/PartitionedCallPartitionedCall,batchnorm_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????@**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_maxpool2d_1_layer_call_and_return_conditional_losses_253852
maxpool2d_1/PartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall$maxpool2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????@**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_263602#
!dropout_1/StatefulPartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_254042"
 conv2d_3/StatefulPartitionedCall?
#batchnorm_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*batchnorm_3_statefulpartitionedcall_args_1*batchnorm_3_statefulpartitionedcall_args_2*batchnorm_3_statefulpartitionedcall_args_3*batchnorm_3_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_3_layer_call_and_return_conditional_losses_264142%
#batchnorm_3/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall,batchnorm_3/StatefulPartitionedCall:output:0'conv2d_4_statefulpartitionedcall_args_1'conv2d_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_255572"
 conv2d_4/StatefulPartitionedCall?
#batchnorm_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*batchnorm_4_statefulpartitionedcall_args_1*batchnorm_4_statefulpartitionedcall_args_2*batchnorm_4_statefulpartitionedcall_args_3*batchnorm_4_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_4_layer_call_and_return_conditional_losses_264962%
#batchnorm_4/StatefulPartitionedCall?
maxpool2d_2/PartitionedCallPartitionedCall,batchnorm_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_maxpool2d_2_layer_call_and_return_conditional_losses_257032
maxpool2d_2/PartitionedCall?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall$maxpool2d_2/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_265632#
!dropout_2/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0'conv2d_5_statefulpartitionedcall_args_1'conv2d_5_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_257222"
 conv2d_5/StatefulPartitionedCall?
#batchnorm_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*batchnorm_5_statefulpartitionedcall_args_1*batchnorm_5_statefulpartitionedcall_args_2*batchnorm_5_statefulpartitionedcall_args_3*batchnorm_5_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_5_layer_call_and_return_conditional_losses_266172%
#batchnorm_5/StatefulPartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall,batchnorm_5/StatefulPartitionedCall:output:0'conv2d_6_statefulpartitionedcall_args_1'conv2d_6_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_258752"
 conv2d_6/StatefulPartitionedCall?
#batchnorm_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*batchnorm_6_statefulpartitionedcall_args_1*batchnorm_6_statefulpartitionedcall_args_2*batchnorm_6_statefulpartitionedcall_args_3*batchnorm_6_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_6_layer_call_and_return_conditional_losses_266992%
#batchnorm_6/StatefulPartitionedCall?
maxpool2d_3/PartitionedCallPartitionedCall,batchnorm_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_maxpool2d_3_layer_call_and_return_conditional_losses_260212
maxpool2d_3/PartitionedCall?
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall$maxpool2d_3/PartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_267662#
!dropout_3/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall*dropout_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????H**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_267902
flatten/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_268092!
dense_1/StatefulPartitionedCall?
#batchnorm_7/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0*batchnorm_7_statefulpartitionedcall_args_1*batchnorm_7_statefulpartitionedcall_args_2*batchnorm_7_statefulpartitionedcall_args_3*batchnorm_7_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_7_layer_call_and_return_conditional_losses_261322%
#batchnorm_7/StatefulPartitionedCall?
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall,batchnorm_7/StatefulPartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_268642#
!dropout_4/StatefulPartitionedCall?
!out_layer/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0(out_layer_statefulpartitionedcall_args_1(out_layer_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_out_layer_layer_call_and_return_conditional_losses_268932#
!out_layer/StatefulPartitionedCall?
IdentityIdentity*out_layer/StatefulPartitionedCall:output:0$^batchnorm_1/StatefulPartitionedCall$^batchnorm_2/StatefulPartitionedCall$^batchnorm_3/StatefulPartitionedCall$^batchnorm_4/StatefulPartitionedCall$^batchnorm_5/StatefulPartitionedCall$^batchnorm_6/StatefulPartitionedCall$^batchnorm_7/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^out_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????00::::::::::::::::::::::::::::::::::::::::::::2J
#batchnorm_1/StatefulPartitionedCall#batchnorm_1/StatefulPartitionedCall2J
#batchnorm_2/StatefulPartitionedCall#batchnorm_2/StatefulPartitionedCall2J
#batchnorm_3/StatefulPartitionedCall#batchnorm_3/StatefulPartitionedCall2J
#batchnorm_4/StatefulPartitionedCall#batchnorm_4/StatefulPartitionedCall2J
#batchnorm_5/StatefulPartitionedCall#batchnorm_5/StatefulPartitionedCall2J
#batchnorm_6/StatefulPartitionedCall#batchnorm_6/StatefulPartitionedCall2J
#batchnorm_7/StatefulPartitionedCall#batchnorm_7/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!out_layer/StatefulPartitionedCall!out_layer/StatefulPartitionedCall:. *
(
_user_specified_nameconv2d_1_input
?
?
+__inference_batchnorm_2_layer_call_fn_28333

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????00@**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_2_layer_call_and_return_conditional_losses_262932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????00@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????00@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
F__inference_batchnorm_6_layer_call_and_return_conditional_losses_26008

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
Const?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?
?
+__inference_batchnorm_6_layer_call_fn_28969

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_6_layer_call_and_return_conditional_losses_259772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
E
)__inference_dropout_3_layer_call_fn_29087

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_267712
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_29093

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? $  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????H2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????H2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
b
F__inference_maxpool2d_2_layer_call_and_return_conditional_losses_25703

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?
?
F__inference_batchnorm_4_layer_call_and_return_conditional_losses_28679

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
Const?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?#
?
F__inference_batchnorm_5_layer_call_and_return_conditional_losses_26617

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_26602
assignmovingavg_1_26609
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *?p}?2	
Const_2?
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/26602*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/26602*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_26602*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/26602*
_output_shapes	
:?2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/26602*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_26602AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/26602*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/26609*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/26609*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_26609*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/26609*
_output_shapes	
:?2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/26609*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_26609AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/26609*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?
?
(__inference_conv2d_4_layer_call_fn_25565

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_255572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?$
?
F__inference_batchnorm_5_layer_call_and_return_conditional_losses_25824

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_25809
assignmovingavg_1_25816
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *?p}?2	
Const_2?
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/25809*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/25809*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_25809*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/25809*
_output_shapes	
:?2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/25809*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_25809AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/25809*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/25816*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/25816*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_25816*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/25816*
_output_shapes	
:?2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/25816*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_25816AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/25816*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_26568

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
?
+__inference_batchnorm_5_layer_call_fn_28818

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_5_layer_call_and_return_conditional_losses_266392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?x
?
?__inference_DCNN_layer_call_and_return_conditional_losses_27171

inputs+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2.
*batchnorm_1_statefulpartitionedcall_args_1.
*batchnorm_1_statefulpartitionedcall_args_2.
*batchnorm_1_statefulpartitionedcall_args_3.
*batchnorm_1_statefulpartitionedcall_args_4+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2.
*batchnorm_2_statefulpartitionedcall_args_1.
*batchnorm_2_statefulpartitionedcall_args_2.
*batchnorm_2_statefulpartitionedcall_args_3.
*batchnorm_2_statefulpartitionedcall_args_4+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_2.
*batchnorm_3_statefulpartitionedcall_args_1.
*batchnorm_3_statefulpartitionedcall_args_2.
*batchnorm_3_statefulpartitionedcall_args_3.
*batchnorm_3_statefulpartitionedcall_args_4+
'conv2d_4_statefulpartitionedcall_args_1+
'conv2d_4_statefulpartitionedcall_args_2.
*batchnorm_4_statefulpartitionedcall_args_1.
*batchnorm_4_statefulpartitionedcall_args_2.
*batchnorm_4_statefulpartitionedcall_args_3.
*batchnorm_4_statefulpartitionedcall_args_4+
'conv2d_5_statefulpartitionedcall_args_1+
'conv2d_5_statefulpartitionedcall_args_2.
*batchnorm_5_statefulpartitionedcall_args_1.
*batchnorm_5_statefulpartitionedcall_args_2.
*batchnorm_5_statefulpartitionedcall_args_3.
*batchnorm_5_statefulpartitionedcall_args_4+
'conv2d_6_statefulpartitionedcall_args_1+
'conv2d_6_statefulpartitionedcall_args_2.
*batchnorm_6_statefulpartitionedcall_args_1.
*batchnorm_6_statefulpartitionedcall_args_2.
*batchnorm_6_statefulpartitionedcall_args_3.
*batchnorm_6_statefulpartitionedcall_args_4*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2.
*batchnorm_7_statefulpartitionedcall_args_1.
*batchnorm_7_statefulpartitionedcall_args_2.
*batchnorm_7_statefulpartitionedcall_args_3.
*batchnorm_7_statefulpartitionedcall_args_4,
(out_layer_statefulpartitionedcall_args_1,
(out_layer_statefulpartitionedcall_args_2
identity??#batchnorm_1/StatefulPartitionedCall?#batchnorm_2/StatefulPartitionedCall?#batchnorm_3/StatefulPartitionedCall?#batchnorm_4/StatefulPartitionedCall?#batchnorm_5/StatefulPartitionedCall?#batchnorm_6/StatefulPartitionedCall?#batchnorm_7/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?!out_layer/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputs'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????00@**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_250862"
 conv2d_1/StatefulPartitionedCall?
#batchnorm_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*batchnorm_1_statefulpartitionedcall_args_1*batchnorm_1_statefulpartitionedcall_args_2*batchnorm_1_statefulpartitionedcall_args_3*batchnorm_1_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????00@**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_1_layer_call_and_return_conditional_losses_262332%
#batchnorm_1/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall,batchnorm_1/StatefulPartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????00@**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_252392"
 conv2d_2/StatefulPartitionedCall?
#batchnorm_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*batchnorm_2_statefulpartitionedcall_args_1*batchnorm_2_statefulpartitionedcall_args_2*batchnorm_2_statefulpartitionedcall_args_3*batchnorm_2_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????00@**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_2_layer_call_and_return_conditional_losses_263152%
#batchnorm_2/StatefulPartitionedCall?
maxpool2d_1/PartitionedCallPartitionedCall,batchnorm_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????@**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_maxpool2d_1_layer_call_and_return_conditional_losses_253852
maxpool2d_1/PartitionedCall?
dropout_1/PartitionedCallPartitionedCall$maxpool2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????@**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_263652
dropout_1/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_254042"
 conv2d_3/StatefulPartitionedCall?
#batchnorm_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*batchnorm_3_statefulpartitionedcall_args_1*batchnorm_3_statefulpartitionedcall_args_2*batchnorm_3_statefulpartitionedcall_args_3*batchnorm_3_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_3_layer_call_and_return_conditional_losses_264362%
#batchnorm_3/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall,batchnorm_3/StatefulPartitionedCall:output:0'conv2d_4_statefulpartitionedcall_args_1'conv2d_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_255572"
 conv2d_4/StatefulPartitionedCall?
#batchnorm_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*batchnorm_4_statefulpartitionedcall_args_1*batchnorm_4_statefulpartitionedcall_args_2*batchnorm_4_statefulpartitionedcall_args_3*batchnorm_4_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_4_layer_call_and_return_conditional_losses_265182%
#batchnorm_4/StatefulPartitionedCall?
maxpool2d_2/PartitionedCallPartitionedCall,batchnorm_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_maxpool2d_2_layer_call_and_return_conditional_losses_257032
maxpool2d_2/PartitionedCall?
dropout_2/PartitionedCallPartitionedCall$maxpool2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_265682
dropout_2/PartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0'conv2d_5_statefulpartitionedcall_args_1'conv2d_5_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_257222"
 conv2d_5/StatefulPartitionedCall?
#batchnorm_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*batchnorm_5_statefulpartitionedcall_args_1*batchnorm_5_statefulpartitionedcall_args_2*batchnorm_5_statefulpartitionedcall_args_3*batchnorm_5_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_5_layer_call_and_return_conditional_losses_266392%
#batchnorm_5/StatefulPartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall,batchnorm_5/StatefulPartitionedCall:output:0'conv2d_6_statefulpartitionedcall_args_1'conv2d_6_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_258752"
 conv2d_6/StatefulPartitionedCall?
#batchnorm_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*batchnorm_6_statefulpartitionedcall_args_1*batchnorm_6_statefulpartitionedcall_args_2*batchnorm_6_statefulpartitionedcall_args_3*batchnorm_6_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_6_layer_call_and_return_conditional_losses_267212%
#batchnorm_6/StatefulPartitionedCall?
maxpool2d_3/PartitionedCallPartitionedCall,batchnorm_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_maxpool2d_3_layer_call_and_return_conditional_losses_260212
maxpool2d_3/PartitionedCall?
dropout_3/PartitionedCallPartitionedCall$maxpool2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_267712
dropout_3/PartitionedCall?
flatten/PartitionedCallPartitionedCall"dropout_3/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????H**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_267902
flatten/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_268092!
dense_1/StatefulPartitionedCall?
#batchnorm_7/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0*batchnorm_7_statefulpartitionedcall_args_1*batchnorm_7_statefulpartitionedcall_args_2*batchnorm_7_statefulpartitionedcall_args_3*batchnorm_7_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_7_layer_call_and_return_conditional_losses_261642%
#batchnorm_7/StatefulPartitionedCall?
dropout_4/PartitionedCallPartitionedCall,batchnorm_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_268692
dropout_4/PartitionedCall?
!out_layer/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0(out_layer_statefulpartitionedcall_args_1(out_layer_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_out_layer_layer_call_and_return_conditional_losses_268932#
!out_layer/StatefulPartitionedCall?
IdentityIdentity*out_layer/StatefulPartitionedCall:output:0$^batchnorm_1/StatefulPartitionedCall$^batchnorm_2/StatefulPartitionedCall$^batchnorm_3/StatefulPartitionedCall$^batchnorm_4/StatefulPartitionedCall$^batchnorm_5/StatefulPartitionedCall$^batchnorm_6/StatefulPartitionedCall$^batchnorm_7/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^out_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????00::::::::::::::::::::::::::::::::::::::::::::2J
#batchnorm_1/StatefulPartitionedCall#batchnorm_1/StatefulPartitionedCall2J
#batchnorm_2/StatefulPartitionedCall#batchnorm_2/StatefulPartitionedCall2J
#batchnorm_3/StatefulPartitionedCall#batchnorm_3/StatefulPartitionedCall2J
#batchnorm_4/StatefulPartitionedCall#batchnorm_4/StatefulPartitionedCall2J
#batchnorm_5/StatefulPartitionedCall#batchnorm_5/StatefulPartitionedCall2J
#batchnorm_6/StatefulPartitionedCall#batchnorm_6/StatefulPartitionedCall2J
#batchnorm_7/StatefulPartitionedCall#batchnorm_7/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!out_layer/StatefulPartitionedCall!out_layer/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
?	
?
B__inference_dense_1_layer_call_and_return_conditional_losses_26809

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?H?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddV
EluEluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Elu?
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????H::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
F__inference_batchnorm_7_layer_call_and_return_conditional_losses_26164

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
E
)__inference_dropout_4_layer_call_fn_29267

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_268692
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
?
(__inference_conv2d_5_layer_call_fn_25730

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_257222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
F__inference_batchnorm_3_layer_call_and_return_conditional_losses_26436

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
Const?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?
?
+__inference_batchnorm_3_layer_call_fn_28537

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_3_layer_call_and_return_conditional_losses_264362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
+__inference_batchnorm_6_layer_call_fn_29043

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_6_layer_call_and_return_conditional_losses_266992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_29257

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
?
+__inference_batchnorm_1_layer_call_fn_28108

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_1_layer_call_and_return_conditional_losses_252192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
F__inference_batchnorm_3_layer_call_and_return_conditional_losses_28519

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
Const?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?
c
D__inference_dropout_4_layer_call_and_return_conditional_losses_29252

inputs
identity?a
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *???2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/random_uniform/max?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniform?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:??????????2
dropout/random_uniform/mul?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:??????????2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqualq
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:??????????2
dropout/mul?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/mul_1f
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
?
F__inference_batchnorm_3_layer_call_and_return_conditional_losses_25537

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
Const?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?$
?
F__inference_batchnorm_6_layer_call_and_return_conditional_losses_28938

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_28923
assignmovingavg_1_28930
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *?p}?2	
Const_2?
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/28923*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/28923*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_28923*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/28923*
_output_shapes	
:?2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/28923*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_28923AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/28923*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/28930*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/28930*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_28930*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/28930*
_output_shapes	
:?2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/28930*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_28930AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/28930*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?#
?
F__inference_batchnorm_2_layer_call_and_return_conditional_losses_26293

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_26278
assignmovingavg_1_26285
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:?????????00@:@:@:@:@:*
epsilon%o?:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *?p}?2	
Const_2?
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/26278*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/26278*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_26278*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/26278*
_output_shapes
:@2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/26278*
_output_shapes
:@2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_26278AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/26278*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/26285*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/26285*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_26285*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/26285*
_output_shapes
:@2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/26285*
_output_shapes
:@2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_26285AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/26285*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????00@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????00@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
ȿ
?=
!__inference__traced_restore_29987
file_prefix&
"assignvariableop_conv2d_1_1_kernel&
"assignvariableop_1_conv2d_1_1_bias*
&assignvariableop_2_batchnorm_1_1_gamma)
%assignvariableop_3_batchnorm_1_1_beta0
,assignvariableop_4_batchnorm_1_1_moving_mean4
0assignvariableop_5_batchnorm_1_1_moving_variance(
$assignvariableop_6_conv2d_2_1_kernel&
"assignvariableop_7_conv2d_2_1_bias*
&assignvariableop_8_batchnorm_2_1_gamma)
%assignvariableop_9_batchnorm_2_1_beta1
-assignvariableop_10_batchnorm_2_1_moving_mean5
1assignvariableop_11_batchnorm_2_1_moving_variance)
%assignvariableop_12_conv2d_3_1_kernel'
#assignvariableop_13_conv2d_3_1_bias+
'assignvariableop_14_batchnorm_3_1_gamma*
&assignvariableop_15_batchnorm_3_1_beta1
-assignvariableop_16_batchnorm_3_1_moving_mean5
1assignvariableop_17_batchnorm_3_1_moving_variance)
%assignvariableop_18_conv2d_4_1_kernel'
#assignvariableop_19_conv2d_4_1_bias+
'assignvariableop_20_batchnorm_4_1_gamma*
&assignvariableop_21_batchnorm_4_1_beta1
-assignvariableop_22_batchnorm_4_1_moving_mean5
1assignvariableop_23_batchnorm_4_1_moving_variance)
%assignvariableop_24_conv2d_5_1_kernel'
#assignvariableop_25_conv2d_5_1_bias+
'assignvariableop_26_batchnorm_5_1_gamma*
&assignvariableop_27_batchnorm_5_1_beta1
-assignvariableop_28_batchnorm_5_1_moving_mean5
1assignvariableop_29_batchnorm_5_1_moving_variance)
%assignvariableop_30_conv2d_6_1_kernel'
#assignvariableop_31_conv2d_6_1_bias+
'assignvariableop_32_batchnorm_6_1_gamma*
&assignvariableop_33_batchnorm_6_1_beta1
-assignvariableop_34_batchnorm_6_1_moving_mean5
1assignvariableop_35_batchnorm_6_1_moving_variance(
$assignvariableop_36_dense_1_1_kernel&
"assignvariableop_37_dense_1_1_bias+
'assignvariableop_38_batchnorm_7_1_gamma*
&assignvariableop_39_batchnorm_7_1_beta1
-assignvariableop_40_batchnorm_7_1_moving_mean5
1assignvariableop_41_batchnorm_7_1_moving_variance*
&assignvariableop_42_out_layer_1_kernel(
$assignvariableop_43_out_layer_1_bias!
assignvariableop_44_adam_iter#
assignvariableop_45_adam_beta_1#
assignvariableop_46_adam_beta_2"
assignvariableop_47_adam_decay*
&assignvariableop_48_adam_learning_rate
assignvariableop_49_total
assignvariableop_50_count0
,assignvariableop_51_adam_conv2d_1_1_kernel_m.
*assignvariableop_52_adam_conv2d_1_1_bias_m2
.assignvariableop_53_adam_batchnorm_1_1_gamma_m1
-assignvariableop_54_adam_batchnorm_1_1_beta_m0
,assignvariableop_55_adam_conv2d_2_1_kernel_m.
*assignvariableop_56_adam_conv2d_2_1_bias_m2
.assignvariableop_57_adam_batchnorm_2_1_gamma_m1
-assignvariableop_58_adam_batchnorm_2_1_beta_m0
,assignvariableop_59_adam_conv2d_3_1_kernel_m.
*assignvariableop_60_adam_conv2d_3_1_bias_m2
.assignvariableop_61_adam_batchnorm_3_1_gamma_m1
-assignvariableop_62_adam_batchnorm_3_1_beta_m0
,assignvariableop_63_adam_conv2d_4_1_kernel_m.
*assignvariableop_64_adam_conv2d_4_1_bias_m2
.assignvariableop_65_adam_batchnorm_4_1_gamma_m1
-assignvariableop_66_adam_batchnorm_4_1_beta_m0
,assignvariableop_67_adam_conv2d_5_1_kernel_m.
*assignvariableop_68_adam_conv2d_5_1_bias_m2
.assignvariableop_69_adam_batchnorm_5_1_gamma_m1
-assignvariableop_70_adam_batchnorm_5_1_beta_m0
,assignvariableop_71_adam_conv2d_6_1_kernel_m.
*assignvariableop_72_adam_conv2d_6_1_bias_m2
.assignvariableop_73_adam_batchnorm_6_1_gamma_m1
-assignvariableop_74_adam_batchnorm_6_1_beta_m/
+assignvariableop_75_adam_dense_1_1_kernel_m-
)assignvariableop_76_adam_dense_1_1_bias_m2
.assignvariableop_77_adam_batchnorm_7_1_gamma_m1
-assignvariableop_78_adam_batchnorm_7_1_beta_m1
-assignvariableop_79_adam_out_layer_1_kernel_m/
+assignvariableop_80_adam_out_layer_1_bias_m0
,assignvariableop_81_adam_conv2d_1_1_kernel_v.
*assignvariableop_82_adam_conv2d_1_1_bias_v2
.assignvariableop_83_adam_batchnorm_1_1_gamma_v1
-assignvariableop_84_adam_batchnorm_1_1_beta_v0
,assignvariableop_85_adam_conv2d_2_1_kernel_v.
*assignvariableop_86_adam_conv2d_2_1_bias_v2
.assignvariableop_87_adam_batchnorm_2_1_gamma_v1
-assignvariableop_88_adam_batchnorm_2_1_beta_v0
,assignvariableop_89_adam_conv2d_3_1_kernel_v.
*assignvariableop_90_adam_conv2d_3_1_bias_v2
.assignvariableop_91_adam_batchnorm_3_1_gamma_v1
-assignvariableop_92_adam_batchnorm_3_1_beta_v0
,assignvariableop_93_adam_conv2d_4_1_kernel_v.
*assignvariableop_94_adam_conv2d_4_1_bias_v2
.assignvariableop_95_adam_batchnorm_4_1_gamma_v1
-assignvariableop_96_adam_batchnorm_4_1_beta_v0
,assignvariableop_97_adam_conv2d_5_1_kernel_v.
*assignvariableop_98_adam_conv2d_5_1_bias_v2
.assignvariableop_99_adam_batchnorm_5_1_gamma_v2
.assignvariableop_100_adam_batchnorm_5_1_beta_v1
-assignvariableop_101_adam_conv2d_6_1_kernel_v/
+assignvariableop_102_adam_conv2d_6_1_bias_v3
/assignvariableop_103_adam_batchnorm_6_1_gamma_v2
.assignvariableop_104_adam_batchnorm_6_1_beta_v0
,assignvariableop_105_adam_dense_1_1_kernel_v.
*assignvariableop_106_adam_dense_1_1_bias_v3
/assignvariableop_107_adam_batchnorm_7_1_gamma_v2
.assignvariableop_108_adam_batchnorm_7_1_beta_v2
.assignvariableop_109_adam_out_layer_1_kernel_v0
,assignvariableop_110_adam_out_layer_1_bias_v
identity_112??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_100?AssignVariableOp_101?AssignVariableOp_102?AssignVariableOp_103?AssignVariableOp_104?AssignVariableOp_105?AssignVariableOp_106?AssignVariableOp_107?AssignVariableOp_108?AssignVariableOp_109?AssignVariableOp_11?AssignVariableOp_110?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?AssignVariableOp_99?	RestoreV2?RestoreV2_1?>
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:o*
dtype0*?=
value?=B?=oB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:o*
dtype0*?
value?B?oB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*}
dtypess
q2o	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp"assignvariableop_conv2d_1_1_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp"assignvariableop_1_conv2d_1_1_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp&assignvariableop_2_batchnorm_1_1_gammaIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp%assignvariableop_3_batchnorm_1_1_betaIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp,assignvariableop_4_batchnorm_1_1_moving_meanIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp0assignvariableop_5_batchnorm_1_1_moving_varianceIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp$assignvariableop_6_conv2d_2_1_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp"assignvariableop_7_conv2d_2_1_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp&assignvariableop_8_batchnorm_2_1_gammaIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp%assignvariableop_9_batchnorm_2_1_betaIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp-assignvariableop_10_batchnorm_2_1_moving_meanIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp1assignvariableop_11_batchnorm_2_1_moving_varianceIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp%assignvariableop_12_conv2d_3_1_kernelIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp#assignvariableop_13_conv2d_3_1_biasIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp'assignvariableop_14_batchnorm_3_1_gammaIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp&assignvariableop_15_batchnorm_3_1_betaIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp-assignvariableop_16_batchnorm_3_1_moving_meanIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp1assignvariableop_17_batchnorm_3_1_moving_varianceIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp%assignvariableop_18_conv2d_4_1_kernelIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp#assignvariableop_19_conv2d_4_1_biasIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp'assignvariableop_20_batchnorm_4_1_gammaIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp&assignvariableop_21_batchnorm_4_1_betaIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp-assignvariableop_22_batchnorm_4_1_moving_meanIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp1assignvariableop_23_batchnorm_4_1_moving_varianceIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp%assignvariableop_24_conv2d_5_1_kernelIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp#assignvariableop_25_conv2d_5_1_biasIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp'assignvariableop_26_batchnorm_5_1_gammaIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp&assignvariableop_27_batchnorm_5_1_betaIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp-assignvariableop_28_batchnorm_5_1_moving_meanIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp1assignvariableop_29_batchnorm_5_1_moving_varianceIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp%assignvariableop_30_conv2d_6_1_kernelIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp#assignvariableop_31_conv2d_6_1_biasIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp'assignvariableop_32_batchnorm_6_1_gammaIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp&assignvariableop_33_batchnorm_6_1_betaIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp-assignvariableop_34_batchnorm_6_1_moving_meanIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp1assignvariableop_35_batchnorm_6_1_moving_varianceIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp$assignvariableop_36_dense_1_1_kernelIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp"assignvariableop_37_dense_1_1_biasIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp'assignvariableop_38_batchnorm_7_1_gammaIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp&assignvariableop_39_batchnorm_7_1_betaIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp-assignvariableop_40_batchnorm_7_1_moving_meanIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp1assignvariableop_41_batchnorm_7_1_moving_varianceIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp&assignvariableop_42_out_layer_1_kernelIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp$assignvariableop_43_out_layer_1_biasIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0	*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOpassignvariableop_44_adam_iterIdentity_44:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_44_
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOpassignvariableop_45_adam_beta_1Identity_45:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_45_
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOpassignvariableop_46_adam_beta_2Identity_46:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_46_
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOpassignvariableop_47_adam_decayIdentity_47:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_47_
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp&assignvariableop_48_adam_learning_rateIdentity_48:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_48_
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOpassignvariableop_49_totalIdentity_49:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_49_
Identity_50IdentityRestoreV2:tensors:50*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOpassignvariableop_50_countIdentity_50:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_50_
Identity_51IdentityRestoreV2:tensors:51*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp,assignvariableop_51_adam_conv2d_1_1_kernel_mIdentity_51:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_51_
Identity_52IdentityRestoreV2:tensors:52*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp*assignvariableop_52_adam_conv2d_1_1_bias_mIdentity_52:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_52_
Identity_53IdentityRestoreV2:tensors:53*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp.assignvariableop_53_adam_batchnorm_1_1_gamma_mIdentity_53:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_53_
Identity_54IdentityRestoreV2:tensors:54*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp-assignvariableop_54_adam_batchnorm_1_1_beta_mIdentity_54:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_54_
Identity_55IdentityRestoreV2:tensors:55*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp,assignvariableop_55_adam_conv2d_2_1_kernel_mIdentity_55:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_55_
Identity_56IdentityRestoreV2:tensors:56*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp*assignvariableop_56_adam_conv2d_2_1_bias_mIdentity_56:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_56_
Identity_57IdentityRestoreV2:tensors:57*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp.assignvariableop_57_adam_batchnorm_2_1_gamma_mIdentity_57:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_57_
Identity_58IdentityRestoreV2:tensors:58*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp-assignvariableop_58_adam_batchnorm_2_1_beta_mIdentity_58:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_58_
Identity_59IdentityRestoreV2:tensors:59*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp,assignvariableop_59_adam_conv2d_3_1_kernel_mIdentity_59:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_59_
Identity_60IdentityRestoreV2:tensors:60*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp*assignvariableop_60_adam_conv2d_3_1_bias_mIdentity_60:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_60_
Identity_61IdentityRestoreV2:tensors:61*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp.assignvariableop_61_adam_batchnorm_3_1_gamma_mIdentity_61:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_61_
Identity_62IdentityRestoreV2:tensors:62*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp-assignvariableop_62_adam_batchnorm_3_1_beta_mIdentity_62:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_62_
Identity_63IdentityRestoreV2:tensors:63*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp,assignvariableop_63_adam_conv2d_4_1_kernel_mIdentity_63:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_63_
Identity_64IdentityRestoreV2:tensors:64*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp*assignvariableop_64_adam_conv2d_4_1_bias_mIdentity_64:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_64_
Identity_65IdentityRestoreV2:tensors:65*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp.assignvariableop_65_adam_batchnorm_4_1_gamma_mIdentity_65:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_65_
Identity_66IdentityRestoreV2:tensors:66*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp-assignvariableop_66_adam_batchnorm_4_1_beta_mIdentity_66:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_66_
Identity_67IdentityRestoreV2:tensors:67*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp,assignvariableop_67_adam_conv2d_5_1_kernel_mIdentity_67:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_67_
Identity_68IdentityRestoreV2:tensors:68*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp*assignvariableop_68_adam_conv2d_5_1_bias_mIdentity_68:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_68_
Identity_69IdentityRestoreV2:tensors:69*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp.assignvariableop_69_adam_batchnorm_5_1_gamma_mIdentity_69:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_69_
Identity_70IdentityRestoreV2:tensors:70*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp-assignvariableop_70_adam_batchnorm_5_1_beta_mIdentity_70:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_70_
Identity_71IdentityRestoreV2:tensors:71*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp,assignvariableop_71_adam_conv2d_6_1_kernel_mIdentity_71:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_71_
Identity_72IdentityRestoreV2:tensors:72*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp*assignvariableop_72_adam_conv2d_6_1_bias_mIdentity_72:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_72_
Identity_73IdentityRestoreV2:tensors:73*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp.assignvariableop_73_adam_batchnorm_6_1_gamma_mIdentity_73:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_73_
Identity_74IdentityRestoreV2:tensors:74*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp-assignvariableop_74_adam_batchnorm_6_1_beta_mIdentity_74:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_74_
Identity_75IdentityRestoreV2:tensors:75*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp+assignvariableop_75_adam_dense_1_1_kernel_mIdentity_75:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_75_
Identity_76IdentityRestoreV2:tensors:76*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp)assignvariableop_76_adam_dense_1_1_bias_mIdentity_76:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_76_
Identity_77IdentityRestoreV2:tensors:77*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp.assignvariableop_77_adam_batchnorm_7_1_gamma_mIdentity_77:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_77_
Identity_78IdentityRestoreV2:tensors:78*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp-assignvariableop_78_adam_batchnorm_7_1_beta_mIdentity_78:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_78_
Identity_79IdentityRestoreV2:tensors:79*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp-assignvariableop_79_adam_out_layer_1_kernel_mIdentity_79:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_79_
Identity_80IdentityRestoreV2:tensors:80*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp+assignvariableop_80_adam_out_layer_1_bias_mIdentity_80:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_80_
Identity_81IdentityRestoreV2:tensors:81*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp,assignvariableop_81_adam_conv2d_1_1_kernel_vIdentity_81:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_81_
Identity_82IdentityRestoreV2:tensors:82*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOp*assignvariableop_82_adam_conv2d_1_1_bias_vIdentity_82:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_82_
Identity_83IdentityRestoreV2:tensors:83*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOp.assignvariableop_83_adam_batchnorm_1_1_gamma_vIdentity_83:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_83_
Identity_84IdentityRestoreV2:tensors:84*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOp-assignvariableop_84_adam_batchnorm_1_1_beta_vIdentity_84:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_84_
Identity_85IdentityRestoreV2:tensors:85*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOp,assignvariableop_85_adam_conv2d_2_1_kernel_vIdentity_85:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_85_
Identity_86IdentityRestoreV2:tensors:86*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOp*assignvariableop_86_adam_conv2d_2_1_bias_vIdentity_86:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_86_
Identity_87IdentityRestoreV2:tensors:87*
T0*
_output_shapes
:2
Identity_87?
AssignVariableOp_87AssignVariableOp.assignvariableop_87_adam_batchnorm_2_1_gamma_vIdentity_87:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_87_
Identity_88IdentityRestoreV2:tensors:88*
T0*
_output_shapes
:2
Identity_88?
AssignVariableOp_88AssignVariableOp-assignvariableop_88_adam_batchnorm_2_1_beta_vIdentity_88:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_88_
Identity_89IdentityRestoreV2:tensors:89*
T0*
_output_shapes
:2
Identity_89?
AssignVariableOp_89AssignVariableOp,assignvariableop_89_adam_conv2d_3_1_kernel_vIdentity_89:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_89_
Identity_90IdentityRestoreV2:tensors:90*
T0*
_output_shapes
:2
Identity_90?
AssignVariableOp_90AssignVariableOp*assignvariableop_90_adam_conv2d_3_1_bias_vIdentity_90:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_90_
Identity_91IdentityRestoreV2:tensors:91*
T0*
_output_shapes
:2
Identity_91?
AssignVariableOp_91AssignVariableOp.assignvariableop_91_adam_batchnorm_3_1_gamma_vIdentity_91:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_91_
Identity_92IdentityRestoreV2:tensors:92*
T0*
_output_shapes
:2
Identity_92?
AssignVariableOp_92AssignVariableOp-assignvariableop_92_adam_batchnorm_3_1_beta_vIdentity_92:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_92_
Identity_93IdentityRestoreV2:tensors:93*
T0*
_output_shapes
:2
Identity_93?
AssignVariableOp_93AssignVariableOp,assignvariableop_93_adam_conv2d_4_1_kernel_vIdentity_93:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_93_
Identity_94IdentityRestoreV2:tensors:94*
T0*
_output_shapes
:2
Identity_94?
AssignVariableOp_94AssignVariableOp*assignvariableop_94_adam_conv2d_4_1_bias_vIdentity_94:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_94_
Identity_95IdentityRestoreV2:tensors:95*
T0*
_output_shapes
:2
Identity_95?
AssignVariableOp_95AssignVariableOp.assignvariableop_95_adam_batchnorm_4_1_gamma_vIdentity_95:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_95_
Identity_96IdentityRestoreV2:tensors:96*
T0*
_output_shapes
:2
Identity_96?
AssignVariableOp_96AssignVariableOp-assignvariableop_96_adam_batchnorm_4_1_beta_vIdentity_96:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_96_
Identity_97IdentityRestoreV2:tensors:97*
T0*
_output_shapes
:2
Identity_97?
AssignVariableOp_97AssignVariableOp,assignvariableop_97_adam_conv2d_5_1_kernel_vIdentity_97:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_97_
Identity_98IdentityRestoreV2:tensors:98*
T0*
_output_shapes
:2
Identity_98?
AssignVariableOp_98AssignVariableOp*assignvariableop_98_adam_conv2d_5_1_bias_vIdentity_98:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_98_
Identity_99IdentityRestoreV2:tensors:99*
T0*
_output_shapes
:2
Identity_99?
AssignVariableOp_99AssignVariableOp.assignvariableop_99_adam_batchnorm_5_1_gamma_vIdentity_99:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_99b
Identity_100IdentityRestoreV2:tensors:100*
T0*
_output_shapes
:2
Identity_100?
AssignVariableOp_100AssignVariableOp.assignvariableop_100_adam_batchnorm_5_1_beta_vIdentity_100:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_100b
Identity_101IdentityRestoreV2:tensors:101*
T0*
_output_shapes
:2
Identity_101?
AssignVariableOp_101AssignVariableOp-assignvariableop_101_adam_conv2d_6_1_kernel_vIdentity_101:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_101b
Identity_102IdentityRestoreV2:tensors:102*
T0*
_output_shapes
:2
Identity_102?
AssignVariableOp_102AssignVariableOp+assignvariableop_102_adam_conv2d_6_1_bias_vIdentity_102:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_102b
Identity_103IdentityRestoreV2:tensors:103*
T0*
_output_shapes
:2
Identity_103?
AssignVariableOp_103AssignVariableOp/assignvariableop_103_adam_batchnorm_6_1_gamma_vIdentity_103:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_103b
Identity_104IdentityRestoreV2:tensors:104*
T0*
_output_shapes
:2
Identity_104?
AssignVariableOp_104AssignVariableOp.assignvariableop_104_adam_batchnorm_6_1_beta_vIdentity_104:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_104b
Identity_105IdentityRestoreV2:tensors:105*
T0*
_output_shapes
:2
Identity_105?
AssignVariableOp_105AssignVariableOp,assignvariableop_105_adam_dense_1_1_kernel_vIdentity_105:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_105b
Identity_106IdentityRestoreV2:tensors:106*
T0*
_output_shapes
:2
Identity_106?
AssignVariableOp_106AssignVariableOp*assignvariableop_106_adam_dense_1_1_bias_vIdentity_106:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_106b
Identity_107IdentityRestoreV2:tensors:107*
T0*
_output_shapes
:2
Identity_107?
AssignVariableOp_107AssignVariableOp/assignvariableop_107_adam_batchnorm_7_1_gamma_vIdentity_107:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_107b
Identity_108IdentityRestoreV2:tensors:108*
T0*
_output_shapes
:2
Identity_108?
AssignVariableOp_108AssignVariableOp.assignvariableop_108_adam_batchnorm_7_1_beta_vIdentity_108:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_108b
Identity_109IdentityRestoreV2:tensors:109*
T0*
_output_shapes
:2
Identity_109?
AssignVariableOp_109AssignVariableOp.assignvariableop_109_adam_out_layer_1_kernel_vIdentity_109:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_109b
Identity_110IdentityRestoreV2:tensors:110*
T0*
_output_shapes
:2
Identity_110?
AssignVariableOp_110AssignVariableOp,assignvariableop_110_adam_out_layer_1_bias_vIdentity_110:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_110?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names?
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_111Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_111?
Identity_112IdentityIdentity_111:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_112"%
identity_112Identity_112:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_992
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
?
?
(__inference_conv2d_2_layer_call_fn_25247

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_252392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
G
+__inference_maxpool2d_3_layer_call_fn_26027

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4????????????????????????????????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_maxpool2d_3_layer_call_and_return_conditional_losses_260212
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?
C
'__inference_flatten_layer_call_fn_29098

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????H**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_267902
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????H2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
b
)__inference_dropout_1_layer_call_fn_28372

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????@**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_263602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?x
?
?__inference_DCNN_layer_call_and_return_conditional_losses_26977
conv2d_1_input+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2.
*batchnorm_1_statefulpartitionedcall_args_1.
*batchnorm_1_statefulpartitionedcall_args_2.
*batchnorm_1_statefulpartitionedcall_args_3.
*batchnorm_1_statefulpartitionedcall_args_4+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2.
*batchnorm_2_statefulpartitionedcall_args_1.
*batchnorm_2_statefulpartitionedcall_args_2.
*batchnorm_2_statefulpartitionedcall_args_3.
*batchnorm_2_statefulpartitionedcall_args_4+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_2.
*batchnorm_3_statefulpartitionedcall_args_1.
*batchnorm_3_statefulpartitionedcall_args_2.
*batchnorm_3_statefulpartitionedcall_args_3.
*batchnorm_3_statefulpartitionedcall_args_4+
'conv2d_4_statefulpartitionedcall_args_1+
'conv2d_4_statefulpartitionedcall_args_2.
*batchnorm_4_statefulpartitionedcall_args_1.
*batchnorm_4_statefulpartitionedcall_args_2.
*batchnorm_4_statefulpartitionedcall_args_3.
*batchnorm_4_statefulpartitionedcall_args_4+
'conv2d_5_statefulpartitionedcall_args_1+
'conv2d_5_statefulpartitionedcall_args_2.
*batchnorm_5_statefulpartitionedcall_args_1.
*batchnorm_5_statefulpartitionedcall_args_2.
*batchnorm_5_statefulpartitionedcall_args_3.
*batchnorm_5_statefulpartitionedcall_args_4+
'conv2d_6_statefulpartitionedcall_args_1+
'conv2d_6_statefulpartitionedcall_args_2.
*batchnorm_6_statefulpartitionedcall_args_1.
*batchnorm_6_statefulpartitionedcall_args_2.
*batchnorm_6_statefulpartitionedcall_args_3.
*batchnorm_6_statefulpartitionedcall_args_4*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2.
*batchnorm_7_statefulpartitionedcall_args_1.
*batchnorm_7_statefulpartitionedcall_args_2.
*batchnorm_7_statefulpartitionedcall_args_3.
*batchnorm_7_statefulpartitionedcall_args_4,
(out_layer_statefulpartitionedcall_args_1,
(out_layer_statefulpartitionedcall_args_2
identity??#batchnorm_1/StatefulPartitionedCall?#batchnorm_2/StatefulPartitionedCall?#batchnorm_3/StatefulPartitionedCall?#batchnorm_4/StatefulPartitionedCall?#batchnorm_5/StatefulPartitionedCall?#batchnorm_6/StatefulPartitionedCall?#batchnorm_7/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?!out_layer/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallconv2d_1_input'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????00@**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_250862"
 conv2d_1/StatefulPartitionedCall?
#batchnorm_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*batchnorm_1_statefulpartitionedcall_args_1*batchnorm_1_statefulpartitionedcall_args_2*batchnorm_1_statefulpartitionedcall_args_3*batchnorm_1_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????00@**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_1_layer_call_and_return_conditional_losses_262332%
#batchnorm_1/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall,batchnorm_1/StatefulPartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????00@**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_252392"
 conv2d_2/StatefulPartitionedCall?
#batchnorm_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*batchnorm_2_statefulpartitionedcall_args_1*batchnorm_2_statefulpartitionedcall_args_2*batchnorm_2_statefulpartitionedcall_args_3*batchnorm_2_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????00@**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_2_layer_call_and_return_conditional_losses_263152%
#batchnorm_2/StatefulPartitionedCall?
maxpool2d_1/PartitionedCallPartitionedCall,batchnorm_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????@**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_maxpool2d_1_layer_call_and_return_conditional_losses_253852
maxpool2d_1/PartitionedCall?
dropout_1/PartitionedCallPartitionedCall$maxpool2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????@**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_263652
dropout_1/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_254042"
 conv2d_3/StatefulPartitionedCall?
#batchnorm_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*batchnorm_3_statefulpartitionedcall_args_1*batchnorm_3_statefulpartitionedcall_args_2*batchnorm_3_statefulpartitionedcall_args_3*batchnorm_3_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_3_layer_call_and_return_conditional_losses_264362%
#batchnorm_3/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall,batchnorm_3/StatefulPartitionedCall:output:0'conv2d_4_statefulpartitionedcall_args_1'conv2d_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_255572"
 conv2d_4/StatefulPartitionedCall?
#batchnorm_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*batchnorm_4_statefulpartitionedcall_args_1*batchnorm_4_statefulpartitionedcall_args_2*batchnorm_4_statefulpartitionedcall_args_3*batchnorm_4_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_4_layer_call_and_return_conditional_losses_265182%
#batchnorm_4/StatefulPartitionedCall?
maxpool2d_2/PartitionedCallPartitionedCall,batchnorm_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_maxpool2d_2_layer_call_and_return_conditional_losses_257032
maxpool2d_2/PartitionedCall?
dropout_2/PartitionedCallPartitionedCall$maxpool2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_265682
dropout_2/PartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0'conv2d_5_statefulpartitionedcall_args_1'conv2d_5_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_257222"
 conv2d_5/StatefulPartitionedCall?
#batchnorm_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*batchnorm_5_statefulpartitionedcall_args_1*batchnorm_5_statefulpartitionedcall_args_2*batchnorm_5_statefulpartitionedcall_args_3*batchnorm_5_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_5_layer_call_and_return_conditional_losses_266392%
#batchnorm_5/StatefulPartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall,batchnorm_5/StatefulPartitionedCall:output:0'conv2d_6_statefulpartitionedcall_args_1'conv2d_6_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_258752"
 conv2d_6/StatefulPartitionedCall?
#batchnorm_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*batchnorm_6_statefulpartitionedcall_args_1*batchnorm_6_statefulpartitionedcall_args_2*batchnorm_6_statefulpartitionedcall_args_3*batchnorm_6_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_6_layer_call_and_return_conditional_losses_267212%
#batchnorm_6/StatefulPartitionedCall?
maxpool2d_3/PartitionedCallPartitionedCall,batchnorm_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_maxpool2d_3_layer_call_and_return_conditional_losses_260212
maxpool2d_3/PartitionedCall?
dropout_3/PartitionedCallPartitionedCall$maxpool2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_267712
dropout_3/PartitionedCall?
flatten/PartitionedCallPartitionedCall"dropout_3/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????H**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_267902
flatten/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_268092!
dense_1/StatefulPartitionedCall?
#batchnorm_7/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0*batchnorm_7_statefulpartitionedcall_args_1*batchnorm_7_statefulpartitionedcall_args_2*batchnorm_7_statefulpartitionedcall_args_3*batchnorm_7_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_7_layer_call_and_return_conditional_losses_261642%
#batchnorm_7/StatefulPartitionedCall?
dropout_4/PartitionedCallPartitionedCall,batchnorm_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_268692
dropout_4/PartitionedCall?
!out_layer/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0(out_layer_statefulpartitionedcall_args_1(out_layer_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_out_layer_layer_call_and_return_conditional_losses_268932#
!out_layer/StatefulPartitionedCall?
IdentityIdentity*out_layer/StatefulPartitionedCall:output:0$^batchnorm_1/StatefulPartitionedCall$^batchnorm_2/StatefulPartitionedCall$^batchnorm_3/StatefulPartitionedCall$^batchnorm_4/StatefulPartitionedCall$^batchnorm_5/StatefulPartitionedCall$^batchnorm_6/StatefulPartitionedCall$^batchnorm_7/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^out_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????00::::::::::::::::::::::::::::::::::::::::::::2J
#batchnorm_1/StatefulPartitionedCall#batchnorm_1/StatefulPartitionedCall2J
#batchnorm_2/StatefulPartitionedCall#batchnorm_2/StatefulPartitionedCall2J
#batchnorm_3/StatefulPartitionedCall#batchnorm_3/StatefulPartitionedCall2J
#batchnorm_4/StatefulPartitionedCall#batchnorm_4/StatefulPartitionedCall2J
#batchnorm_5/StatefulPartitionedCall#batchnorm_5/StatefulPartitionedCall2J
#batchnorm_6/StatefulPartitionedCall#batchnorm_6/StatefulPartitionedCall2J
#batchnorm_7/StatefulPartitionedCall#batchnorm_7/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!out_layer/StatefulPartitionedCall!out_layer/StatefulPartitionedCall:. *
(
_user_specified_nameconv2d_1_input
?
G
+__inference_maxpool2d_1_layer_call_fn_25391

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4????????????????????????????????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_maxpool2d_1_layer_call_and_return_conditional_losses_253852
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?
?
(__inference_conv2d_6_layer_call_fn_25883

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_258752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?$
?
F__inference_batchnorm_4_layer_call_and_return_conditional_losses_28583

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_28568
assignmovingavg_1_28575
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *?p}?2	
Const_2?
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/28568*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/28568*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_28568*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/28568*
_output_shapes	
:?2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/28568*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_28568AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/28568*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/28575*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/28575*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_28575*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/28575*
_output_shapes	
:?2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/28575*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_28575AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/28575*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?$
?
F__inference_batchnorm_1_layer_call_and_return_conditional_losses_25188

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_25173
assignmovingavg_1_25180
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *?p}?2	
Const_2?
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/25173*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/25173*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_25173*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/25173*
_output_shapes
:@2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/25173*
_output_shapes
:@2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_25173AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/25173*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/25180*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/25180*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_25180*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/25180*
_output_shapes
:@2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/25180*
_output_shapes
:@2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_25180AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/25180*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?
?
F__inference_batchnorm_2_layer_call_and_return_conditional_losses_28324

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
Const?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????00@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????00@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?
?
F__inference_batchnorm_6_layer_call_and_return_conditional_losses_26721

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
Const?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_26563

inputs
identity?a
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/random_uniform/max?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniform?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*0
_output_shapes
:??????????2
dropout/random_uniform/mul?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*0
_output_shapes
:??????????2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*0
_output_shapes
:??????????2
dropout/GreaterEqualy
dropout/mulMulinputsdropout/truediv:z:0*
T0*0
_output_shapes
:??????????2
dropout/mul?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
dropout/Cast?
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
dropout/mul_1n
IdentityIdentitydropout/mul_1:z:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_25404

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAddp
EluEluBiasAdd:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
Elu?
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_26365

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????@:& "
 
_user_specified_nameinputs
?#
?
F__inference_batchnorm_1_layer_call_and_return_conditional_losses_26211

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_26196
assignmovingavg_1_26203
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:?????????00@:@:@:@:@:*
epsilon%o?:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *?p}?2	
Const_2?
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/26196*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/26196*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_26196*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/26196*
_output_shapes
:@2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/26196*
_output_shapes
:@2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_26196AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/26196*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/26203*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/26203*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_26203*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/26203*
_output_shapes
:@2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/26203*
_output_shapes
:@2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_26203AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/26203*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????00@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????00@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?
?
F__inference_batchnorm_2_layer_call_and_return_conditional_losses_25372

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
Const?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?
b
F__inference_maxpool2d_1_layer_call_and_return_conditional_losses_25385

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?
?
F__inference_batchnorm_5_layer_call_and_return_conditional_losses_28874

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
Const?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?
?
F__inference_batchnorm_1_layer_call_and_return_conditional_losses_25219

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
Const?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?
?
$__inference_DCNN_layer_call_fn_27973

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32#
statefulpartitionedcall_args_33#
statefulpartitionedcall_args_34#
statefulpartitionedcall_args_35#
statefulpartitionedcall_args_36#
statefulpartitionedcall_args_37#
statefulpartitionedcall_args_38#
statefulpartitionedcall_args_39#
statefulpartitionedcall_args_40#
statefulpartitionedcall_args_41#
statefulpartitionedcall_args_42#
statefulpartitionedcall_args_43#
statefulpartitionedcall_args_44
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32statefulpartitionedcall_args_33statefulpartitionedcall_args_34statefulpartitionedcall_args_35statefulpartitionedcall_args_36statefulpartitionedcall_args_37statefulpartitionedcall_args_38statefulpartitionedcall_args_39statefulpartitionedcall_args_40statefulpartitionedcall_args_41statefulpartitionedcall_args_42statefulpartitionedcall_args_43statefulpartitionedcall_args_44*8
Tin1
/2-*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_DCNN_layer_call_and_return_conditional_losses_270512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????00::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
F__inference_batchnorm_2_layer_call_and_return_conditional_losses_26315

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
Const?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????00@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????00@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?
?
F__inference_batchnorm_1_layer_call_and_return_conditional_losses_28090

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
Const?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?
b
)__inference_dropout_3_layer_call_fn_29082

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_267662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?$
?
F__inference_batchnorm_2_layer_call_and_return_conditional_losses_28228

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_28213
assignmovingavg_1_28220
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *?p}?2	
Const_2?
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/28213*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/28213*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_28213*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/28213*
_output_shapes
:@2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/28213*
_output_shapes
:@2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_28213AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/28213*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/28220*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/28220*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_28220*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/28220*
_output_shapes
:@2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/28220*
_output_shapes
:@2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_28220AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/28220*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?
?
$__inference_DCNN_layer_call_fn_28022

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32#
statefulpartitionedcall_args_33#
statefulpartitionedcall_args_34#
statefulpartitionedcall_args_35#
statefulpartitionedcall_args_36#
statefulpartitionedcall_args_37#
statefulpartitionedcall_args_38#
statefulpartitionedcall_args_39#
statefulpartitionedcall_args_40#
statefulpartitionedcall_args_41#
statefulpartitionedcall_args_42#
statefulpartitionedcall_args_43#
statefulpartitionedcall_args_44
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32statefulpartitionedcall_args_33statefulpartitionedcall_args_34statefulpartitionedcall_args_35statefulpartitionedcall_args_36statefulpartitionedcall_args_37statefulpartitionedcall_args_38statefulpartitionedcall_args_39statefulpartitionedcall_args_40statefulpartitionedcall_args_41statefulpartitionedcall_args_42statefulpartitionedcall_args_43statefulpartitionedcall_args_44*8
Tin1
/2-*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_DCNN_layer_call_and_return_conditional_losses_271712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????00::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ׁ
?!
 __inference__wrapped_model_25073
conv2d_1_input0
,dcnn_conv2d_1_conv2d_readvariableop_resource1
-dcnn_conv2d_1_biasadd_readvariableop_resource,
(dcnn_batchnorm_1_readvariableop_resource.
*dcnn_batchnorm_1_readvariableop_1_resource=
9dcnn_batchnorm_1_fusedbatchnormv3_readvariableop_resource?
;dcnn_batchnorm_1_fusedbatchnormv3_readvariableop_1_resource0
,dcnn_conv2d_2_conv2d_readvariableop_resource1
-dcnn_conv2d_2_biasadd_readvariableop_resource,
(dcnn_batchnorm_2_readvariableop_resource.
*dcnn_batchnorm_2_readvariableop_1_resource=
9dcnn_batchnorm_2_fusedbatchnormv3_readvariableop_resource?
;dcnn_batchnorm_2_fusedbatchnormv3_readvariableop_1_resource0
,dcnn_conv2d_3_conv2d_readvariableop_resource1
-dcnn_conv2d_3_biasadd_readvariableop_resource,
(dcnn_batchnorm_3_readvariableop_resource.
*dcnn_batchnorm_3_readvariableop_1_resource=
9dcnn_batchnorm_3_fusedbatchnormv3_readvariableop_resource?
;dcnn_batchnorm_3_fusedbatchnormv3_readvariableop_1_resource0
,dcnn_conv2d_4_conv2d_readvariableop_resource1
-dcnn_conv2d_4_biasadd_readvariableop_resource,
(dcnn_batchnorm_4_readvariableop_resource.
*dcnn_batchnorm_4_readvariableop_1_resource=
9dcnn_batchnorm_4_fusedbatchnormv3_readvariableop_resource?
;dcnn_batchnorm_4_fusedbatchnormv3_readvariableop_1_resource0
,dcnn_conv2d_5_conv2d_readvariableop_resource1
-dcnn_conv2d_5_biasadd_readvariableop_resource,
(dcnn_batchnorm_5_readvariableop_resource.
*dcnn_batchnorm_5_readvariableop_1_resource=
9dcnn_batchnorm_5_fusedbatchnormv3_readvariableop_resource?
;dcnn_batchnorm_5_fusedbatchnormv3_readvariableop_1_resource0
,dcnn_conv2d_6_conv2d_readvariableop_resource1
-dcnn_conv2d_6_biasadd_readvariableop_resource,
(dcnn_batchnorm_6_readvariableop_resource.
*dcnn_batchnorm_6_readvariableop_1_resource=
9dcnn_batchnorm_6_fusedbatchnormv3_readvariableop_resource?
;dcnn_batchnorm_6_fusedbatchnormv3_readvariableop_1_resource/
+dcnn_dense_1_matmul_readvariableop_resource0
,dcnn_dense_1_biasadd_readvariableop_resource6
2dcnn_batchnorm_7_batchnorm_readvariableop_resource:
6dcnn_batchnorm_7_batchnorm_mul_readvariableop_resource8
4dcnn_batchnorm_7_batchnorm_readvariableop_1_resource8
4dcnn_batchnorm_7_batchnorm_readvariableop_2_resource1
-dcnn_out_layer_matmul_readvariableop_resource2
.dcnn_out_layer_biasadd_readvariableop_resource
identity??0DCNN/batchnorm_1/FusedBatchNormV3/ReadVariableOp?2DCNN/batchnorm_1/FusedBatchNormV3/ReadVariableOp_1?DCNN/batchnorm_1/ReadVariableOp?!DCNN/batchnorm_1/ReadVariableOp_1?0DCNN/batchnorm_2/FusedBatchNormV3/ReadVariableOp?2DCNN/batchnorm_2/FusedBatchNormV3/ReadVariableOp_1?DCNN/batchnorm_2/ReadVariableOp?!DCNN/batchnorm_2/ReadVariableOp_1?0DCNN/batchnorm_3/FusedBatchNormV3/ReadVariableOp?2DCNN/batchnorm_3/FusedBatchNormV3/ReadVariableOp_1?DCNN/batchnorm_3/ReadVariableOp?!DCNN/batchnorm_3/ReadVariableOp_1?0DCNN/batchnorm_4/FusedBatchNormV3/ReadVariableOp?2DCNN/batchnorm_4/FusedBatchNormV3/ReadVariableOp_1?DCNN/batchnorm_4/ReadVariableOp?!DCNN/batchnorm_4/ReadVariableOp_1?0DCNN/batchnorm_5/FusedBatchNormV3/ReadVariableOp?2DCNN/batchnorm_5/FusedBatchNormV3/ReadVariableOp_1?DCNN/batchnorm_5/ReadVariableOp?!DCNN/batchnorm_5/ReadVariableOp_1?0DCNN/batchnorm_6/FusedBatchNormV3/ReadVariableOp?2DCNN/batchnorm_6/FusedBatchNormV3/ReadVariableOp_1?DCNN/batchnorm_6/ReadVariableOp?!DCNN/batchnorm_6/ReadVariableOp_1?)DCNN/batchnorm_7/batchnorm/ReadVariableOp?+DCNN/batchnorm_7/batchnorm/ReadVariableOp_1?+DCNN/batchnorm_7/batchnorm/ReadVariableOp_2?-DCNN/batchnorm_7/batchnorm/mul/ReadVariableOp?$DCNN/conv2d_1/BiasAdd/ReadVariableOp?#DCNN/conv2d_1/Conv2D/ReadVariableOp?$DCNN/conv2d_2/BiasAdd/ReadVariableOp?#DCNN/conv2d_2/Conv2D/ReadVariableOp?$DCNN/conv2d_3/BiasAdd/ReadVariableOp?#DCNN/conv2d_3/Conv2D/ReadVariableOp?$DCNN/conv2d_4/BiasAdd/ReadVariableOp?#DCNN/conv2d_4/Conv2D/ReadVariableOp?$DCNN/conv2d_5/BiasAdd/ReadVariableOp?#DCNN/conv2d_5/Conv2D/ReadVariableOp?$DCNN/conv2d_6/BiasAdd/ReadVariableOp?#DCNN/conv2d_6/Conv2D/ReadVariableOp?#DCNN/dense_1/BiasAdd/ReadVariableOp?"DCNN/dense_1/MatMul/ReadVariableOp?%DCNN/out_layer/BiasAdd/ReadVariableOp?$DCNN/out_layer/MatMul/ReadVariableOp?
#DCNN/conv2d_1/Conv2D/ReadVariableOpReadVariableOp,dcnn_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02%
#DCNN/conv2d_1/Conv2D/ReadVariableOp?
DCNN/conv2d_1/Conv2DConv2Dconv2d_1_input+DCNN/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00@*
paddingSAME*
strides
2
DCNN/conv2d_1/Conv2D?
$DCNN/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp-dcnn_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02&
$DCNN/conv2d_1/BiasAdd/ReadVariableOp?
DCNN/conv2d_1/BiasAddBiasAddDCNN/conv2d_1/Conv2D:output:0,DCNN/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00@2
DCNN/conv2d_1/BiasAdd?
DCNN/conv2d_1/EluEluDCNN/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00@2
DCNN/conv2d_1/Elu?
DCNN/batchnorm_1/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
DCNN/batchnorm_1/LogicalAnd/x?
DCNN/batchnorm_1/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
DCNN/batchnorm_1/LogicalAnd/y?
DCNN/batchnorm_1/LogicalAnd
LogicalAnd&DCNN/batchnorm_1/LogicalAnd/x:output:0&DCNN/batchnorm_1/LogicalAnd/y:output:0*
_output_shapes
: 2
DCNN/batchnorm_1/LogicalAnd?
DCNN/batchnorm_1/ReadVariableOpReadVariableOp(dcnn_batchnorm_1_readvariableop_resource*
_output_shapes
:@*
dtype02!
DCNN/batchnorm_1/ReadVariableOp?
!DCNN/batchnorm_1/ReadVariableOp_1ReadVariableOp*dcnn_batchnorm_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!DCNN/batchnorm_1/ReadVariableOp_1?
0DCNN/batchnorm_1/FusedBatchNormV3/ReadVariableOpReadVariableOp9dcnn_batchnorm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype022
0DCNN/batchnorm_1/FusedBatchNormV3/ReadVariableOp?
2DCNN/batchnorm_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;dcnn_batchnorm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2DCNN/batchnorm_1/FusedBatchNormV3/ReadVariableOp_1?
!DCNN/batchnorm_1/FusedBatchNormV3FusedBatchNormV3DCNN/conv2d_1/Elu:activations:0'DCNN/batchnorm_1/ReadVariableOp:value:0)DCNN/batchnorm_1/ReadVariableOp_1:value:08DCNN/batchnorm_1/FusedBatchNormV3/ReadVariableOp:value:0:DCNN/batchnorm_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00@:@:@:@:@:*
epsilon%o?:*
is_training( 2#
!DCNN/batchnorm_1/FusedBatchNormV3u
DCNN/batchnorm_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
DCNN/batchnorm_1/Const?
#DCNN/conv2d_2/Conv2D/ReadVariableOpReadVariableOp,dcnn_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02%
#DCNN/conv2d_2/Conv2D/ReadVariableOp?
DCNN/conv2d_2/Conv2DConv2D%DCNN/batchnorm_1/FusedBatchNormV3:y:0+DCNN/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00@*
paddingSAME*
strides
2
DCNN/conv2d_2/Conv2D?
$DCNN/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp-dcnn_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02&
$DCNN/conv2d_2/BiasAdd/ReadVariableOp?
DCNN/conv2d_2/BiasAddBiasAddDCNN/conv2d_2/Conv2D:output:0,DCNN/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00@2
DCNN/conv2d_2/BiasAdd?
DCNN/conv2d_2/EluEluDCNN/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00@2
DCNN/conv2d_2/Elu?
DCNN/batchnorm_2/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
DCNN/batchnorm_2/LogicalAnd/x?
DCNN/batchnorm_2/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
DCNN/batchnorm_2/LogicalAnd/y?
DCNN/batchnorm_2/LogicalAnd
LogicalAnd&DCNN/batchnorm_2/LogicalAnd/x:output:0&DCNN/batchnorm_2/LogicalAnd/y:output:0*
_output_shapes
: 2
DCNN/batchnorm_2/LogicalAnd?
DCNN/batchnorm_2/ReadVariableOpReadVariableOp(dcnn_batchnorm_2_readvariableop_resource*
_output_shapes
:@*
dtype02!
DCNN/batchnorm_2/ReadVariableOp?
!DCNN/batchnorm_2/ReadVariableOp_1ReadVariableOp*dcnn_batchnorm_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!DCNN/batchnorm_2/ReadVariableOp_1?
0DCNN/batchnorm_2/FusedBatchNormV3/ReadVariableOpReadVariableOp9dcnn_batchnorm_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype022
0DCNN/batchnorm_2/FusedBatchNormV3/ReadVariableOp?
2DCNN/batchnorm_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;dcnn_batchnorm_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2DCNN/batchnorm_2/FusedBatchNormV3/ReadVariableOp_1?
!DCNN/batchnorm_2/FusedBatchNormV3FusedBatchNormV3DCNN/conv2d_2/Elu:activations:0'DCNN/batchnorm_2/ReadVariableOp:value:0)DCNN/batchnorm_2/ReadVariableOp_1:value:08DCNN/batchnorm_2/FusedBatchNormV3/ReadVariableOp:value:0:DCNN/batchnorm_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00@:@:@:@:@:*
epsilon%o?:*
is_training( 2#
!DCNN/batchnorm_2/FusedBatchNormV3u
DCNN/batchnorm_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
DCNN/batchnorm_2/Const?
DCNN/maxpool2d_1/MaxPoolMaxPool%DCNN/batchnorm_2/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
DCNN/maxpool2d_1/MaxPool?
DCNN/dropout_1/IdentityIdentity!DCNN/maxpool2d_1/MaxPool:output:0*
T0*/
_output_shapes
:?????????@2
DCNN/dropout_1/Identity?
#DCNN/conv2d_3/Conv2D/ReadVariableOpReadVariableOp,dcnn_conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02%
#DCNN/conv2d_3/Conv2D/ReadVariableOp?
DCNN/conv2d_3/Conv2DConv2D DCNN/dropout_1/Identity:output:0+DCNN/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
DCNN/conv2d_3/Conv2D?
$DCNN/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp-dcnn_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$DCNN/conv2d_3/BiasAdd/ReadVariableOp?
DCNN/conv2d_3/BiasAddBiasAddDCNN/conv2d_3/Conv2D:output:0,DCNN/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
DCNN/conv2d_3/BiasAdd?
DCNN/conv2d_3/EluEluDCNN/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
DCNN/conv2d_3/Elu?
DCNN/batchnorm_3/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
DCNN/batchnorm_3/LogicalAnd/x?
DCNN/batchnorm_3/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
DCNN/batchnorm_3/LogicalAnd/y?
DCNN/batchnorm_3/LogicalAnd
LogicalAnd&DCNN/batchnorm_3/LogicalAnd/x:output:0&DCNN/batchnorm_3/LogicalAnd/y:output:0*
_output_shapes
: 2
DCNN/batchnorm_3/LogicalAnd?
DCNN/batchnorm_3/ReadVariableOpReadVariableOp(dcnn_batchnorm_3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
DCNN/batchnorm_3/ReadVariableOp?
!DCNN/batchnorm_3/ReadVariableOp_1ReadVariableOp*dcnn_batchnorm_3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!DCNN/batchnorm_3/ReadVariableOp_1?
0DCNN/batchnorm_3/FusedBatchNormV3/ReadVariableOpReadVariableOp9dcnn_batchnorm_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype022
0DCNN/batchnorm_3/FusedBatchNormV3/ReadVariableOp?
2DCNN/batchnorm_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;dcnn_batchnorm_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype024
2DCNN/batchnorm_3/FusedBatchNormV3/ReadVariableOp_1?
!DCNN/batchnorm_3/FusedBatchNormV3FusedBatchNormV3DCNN/conv2d_3/Elu:activations:0'DCNN/batchnorm_3/ReadVariableOp:value:0)DCNN/batchnorm_3/ReadVariableOp_1:value:08DCNN/batchnorm_3/FusedBatchNormV3/ReadVariableOp:value:0:DCNN/batchnorm_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2#
!DCNN/batchnorm_3/FusedBatchNormV3u
DCNN/batchnorm_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
DCNN/batchnorm_3/Const?
#DCNN/conv2d_4/Conv2D/ReadVariableOpReadVariableOp,dcnn_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02%
#DCNN/conv2d_4/Conv2D/ReadVariableOp?
DCNN/conv2d_4/Conv2DConv2D%DCNN/batchnorm_3/FusedBatchNormV3:y:0+DCNN/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
DCNN/conv2d_4/Conv2D?
$DCNN/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp-dcnn_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$DCNN/conv2d_4/BiasAdd/ReadVariableOp?
DCNN/conv2d_4/BiasAddBiasAddDCNN/conv2d_4/Conv2D:output:0,DCNN/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
DCNN/conv2d_4/BiasAdd?
DCNN/conv2d_4/EluEluDCNN/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
DCNN/conv2d_4/Elu?
DCNN/batchnorm_4/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
DCNN/batchnorm_4/LogicalAnd/x?
DCNN/batchnorm_4/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
DCNN/batchnorm_4/LogicalAnd/y?
DCNN/batchnorm_4/LogicalAnd
LogicalAnd&DCNN/batchnorm_4/LogicalAnd/x:output:0&DCNN/batchnorm_4/LogicalAnd/y:output:0*
_output_shapes
: 2
DCNN/batchnorm_4/LogicalAnd?
DCNN/batchnorm_4/ReadVariableOpReadVariableOp(dcnn_batchnorm_4_readvariableop_resource*
_output_shapes	
:?*
dtype02!
DCNN/batchnorm_4/ReadVariableOp?
!DCNN/batchnorm_4/ReadVariableOp_1ReadVariableOp*dcnn_batchnorm_4_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!DCNN/batchnorm_4/ReadVariableOp_1?
0DCNN/batchnorm_4/FusedBatchNormV3/ReadVariableOpReadVariableOp9dcnn_batchnorm_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype022
0DCNN/batchnorm_4/FusedBatchNormV3/ReadVariableOp?
2DCNN/batchnorm_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;dcnn_batchnorm_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype024
2DCNN/batchnorm_4/FusedBatchNormV3/ReadVariableOp_1?
!DCNN/batchnorm_4/FusedBatchNormV3FusedBatchNormV3DCNN/conv2d_4/Elu:activations:0'DCNN/batchnorm_4/ReadVariableOp:value:0)DCNN/batchnorm_4/ReadVariableOp_1:value:08DCNN/batchnorm_4/FusedBatchNormV3/ReadVariableOp:value:0:DCNN/batchnorm_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2#
!DCNN/batchnorm_4/FusedBatchNormV3u
DCNN/batchnorm_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
DCNN/batchnorm_4/Const?
DCNN/maxpool2d_2/MaxPoolMaxPool%DCNN/batchnorm_4/FusedBatchNormV3:y:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
DCNN/maxpool2d_2/MaxPool?
DCNN/dropout_2/IdentityIdentity!DCNN/maxpool2d_2/MaxPool:output:0*
T0*0
_output_shapes
:??????????2
DCNN/dropout_2/Identity?
#DCNN/conv2d_5/Conv2D/ReadVariableOpReadVariableOp,dcnn_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02%
#DCNN/conv2d_5/Conv2D/ReadVariableOp?
DCNN/conv2d_5/Conv2DConv2D DCNN/dropout_2/Identity:output:0+DCNN/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
DCNN/conv2d_5/Conv2D?
$DCNN/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp-dcnn_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$DCNN/conv2d_5/BiasAdd/ReadVariableOp?
DCNN/conv2d_5/BiasAddBiasAddDCNN/conv2d_5/Conv2D:output:0,DCNN/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
DCNN/conv2d_5/BiasAdd?
DCNN/conv2d_5/EluEluDCNN/conv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
DCNN/conv2d_5/Elu?
DCNN/batchnorm_5/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
DCNN/batchnorm_5/LogicalAnd/x?
DCNN/batchnorm_5/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
DCNN/batchnorm_5/LogicalAnd/y?
DCNN/batchnorm_5/LogicalAnd
LogicalAnd&DCNN/batchnorm_5/LogicalAnd/x:output:0&DCNN/batchnorm_5/LogicalAnd/y:output:0*
_output_shapes
: 2
DCNN/batchnorm_5/LogicalAnd?
DCNN/batchnorm_5/ReadVariableOpReadVariableOp(dcnn_batchnorm_5_readvariableop_resource*
_output_shapes	
:?*
dtype02!
DCNN/batchnorm_5/ReadVariableOp?
!DCNN/batchnorm_5/ReadVariableOp_1ReadVariableOp*dcnn_batchnorm_5_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!DCNN/batchnorm_5/ReadVariableOp_1?
0DCNN/batchnorm_5/FusedBatchNormV3/ReadVariableOpReadVariableOp9dcnn_batchnorm_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype022
0DCNN/batchnorm_5/FusedBatchNormV3/ReadVariableOp?
2DCNN/batchnorm_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;dcnn_batchnorm_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype024
2DCNN/batchnorm_5/FusedBatchNormV3/ReadVariableOp_1?
!DCNN/batchnorm_5/FusedBatchNormV3FusedBatchNormV3DCNN/conv2d_5/Elu:activations:0'DCNN/batchnorm_5/ReadVariableOp:value:0)DCNN/batchnorm_5/ReadVariableOp_1:value:08DCNN/batchnorm_5/FusedBatchNormV3/ReadVariableOp:value:0:DCNN/batchnorm_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2#
!DCNN/batchnorm_5/FusedBatchNormV3u
DCNN/batchnorm_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
DCNN/batchnorm_5/Const?
#DCNN/conv2d_6/Conv2D/ReadVariableOpReadVariableOp,dcnn_conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02%
#DCNN/conv2d_6/Conv2D/ReadVariableOp?
DCNN/conv2d_6/Conv2DConv2D%DCNN/batchnorm_5/FusedBatchNormV3:y:0+DCNN/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
DCNN/conv2d_6/Conv2D?
$DCNN/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp-dcnn_conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$DCNN/conv2d_6/BiasAdd/ReadVariableOp?
DCNN/conv2d_6/BiasAddBiasAddDCNN/conv2d_6/Conv2D:output:0,DCNN/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
DCNN/conv2d_6/BiasAdd?
DCNN/conv2d_6/EluEluDCNN/conv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
DCNN/conv2d_6/Elu?
DCNN/batchnorm_6/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
DCNN/batchnorm_6/LogicalAnd/x?
DCNN/batchnorm_6/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
DCNN/batchnorm_6/LogicalAnd/y?
DCNN/batchnorm_6/LogicalAnd
LogicalAnd&DCNN/batchnorm_6/LogicalAnd/x:output:0&DCNN/batchnorm_6/LogicalAnd/y:output:0*
_output_shapes
: 2
DCNN/batchnorm_6/LogicalAnd?
DCNN/batchnorm_6/ReadVariableOpReadVariableOp(dcnn_batchnorm_6_readvariableop_resource*
_output_shapes	
:?*
dtype02!
DCNN/batchnorm_6/ReadVariableOp?
!DCNN/batchnorm_6/ReadVariableOp_1ReadVariableOp*dcnn_batchnorm_6_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!DCNN/batchnorm_6/ReadVariableOp_1?
0DCNN/batchnorm_6/FusedBatchNormV3/ReadVariableOpReadVariableOp9dcnn_batchnorm_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype022
0DCNN/batchnorm_6/FusedBatchNormV3/ReadVariableOp?
2DCNN/batchnorm_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;dcnn_batchnorm_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype024
2DCNN/batchnorm_6/FusedBatchNormV3/ReadVariableOp_1?
!DCNN/batchnorm_6/FusedBatchNormV3FusedBatchNormV3DCNN/conv2d_6/Elu:activations:0'DCNN/batchnorm_6/ReadVariableOp:value:0)DCNN/batchnorm_6/ReadVariableOp_1:value:08DCNN/batchnorm_6/FusedBatchNormV3/ReadVariableOp:value:0:DCNN/batchnorm_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2#
!DCNN/batchnorm_6/FusedBatchNormV3u
DCNN/batchnorm_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
DCNN/batchnorm_6/Const?
DCNN/maxpool2d_3/MaxPoolMaxPool%DCNN/batchnorm_6/FusedBatchNormV3:y:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
DCNN/maxpool2d_3/MaxPool?
DCNN/dropout_3/IdentityIdentity!DCNN/maxpool2d_3/MaxPool:output:0*
T0*0
_output_shapes
:??????????2
DCNN/dropout_3/Identityy
DCNN/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? $  2
DCNN/flatten/Const?
DCNN/flatten/ReshapeReshape DCNN/dropout_3/Identity:output:0DCNN/flatten/Const:output:0*
T0*(
_output_shapes
:??????????H2
DCNN/flatten/Reshape?
"DCNN/dense_1/MatMul/ReadVariableOpReadVariableOp+dcnn_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
?H?*
dtype02$
"DCNN/dense_1/MatMul/ReadVariableOp?
DCNN/dense_1/MatMulMatMulDCNN/flatten/Reshape:output:0*DCNN/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
DCNN/dense_1/MatMul?
#DCNN/dense_1/BiasAdd/ReadVariableOpReadVariableOp,dcnn_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#DCNN/dense_1/BiasAdd/ReadVariableOp?
DCNN/dense_1/BiasAddBiasAddDCNN/dense_1/MatMul:product:0+DCNN/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
DCNN/dense_1/BiasAdd}
DCNN/dense_1/EluEluDCNN/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
DCNN/dense_1/Elu?
DCNN/batchnorm_7/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
DCNN/batchnorm_7/LogicalAnd/x?
DCNN/batchnorm_7/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
DCNN/batchnorm_7/LogicalAnd/y?
DCNN/batchnorm_7/LogicalAnd
LogicalAnd&DCNN/batchnorm_7/LogicalAnd/x:output:0&DCNN/batchnorm_7/LogicalAnd/y:output:0*
_output_shapes
: 2
DCNN/batchnorm_7/LogicalAnd?
)DCNN/batchnorm_7/batchnorm/ReadVariableOpReadVariableOp2dcnn_batchnorm_7_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)DCNN/batchnorm_7/batchnorm/ReadVariableOp?
 DCNN/batchnorm_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2"
 DCNN/batchnorm_7/batchnorm/add/y?
DCNN/batchnorm_7/batchnorm/addAddV21DCNN/batchnorm_7/batchnorm/ReadVariableOp:value:0)DCNN/batchnorm_7/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2 
DCNN/batchnorm_7/batchnorm/add?
 DCNN/batchnorm_7/batchnorm/RsqrtRsqrt"DCNN/batchnorm_7/batchnorm/add:z:0*
T0*
_output_shapes	
:?2"
 DCNN/batchnorm_7/batchnorm/Rsqrt?
-DCNN/batchnorm_7/batchnorm/mul/ReadVariableOpReadVariableOp6dcnn_batchnorm_7_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-DCNN/batchnorm_7/batchnorm/mul/ReadVariableOp?
DCNN/batchnorm_7/batchnorm/mulMul$DCNN/batchnorm_7/batchnorm/Rsqrt:y:05DCNN/batchnorm_7/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2 
DCNN/batchnorm_7/batchnorm/mul?
 DCNN/batchnorm_7/batchnorm/mul_1MulDCNN/dense_1/Elu:activations:0"DCNN/batchnorm_7/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2"
 DCNN/batchnorm_7/batchnorm/mul_1?
+DCNN/batchnorm_7/batchnorm/ReadVariableOp_1ReadVariableOp4dcnn_batchnorm_7_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02-
+DCNN/batchnorm_7/batchnorm/ReadVariableOp_1?
 DCNN/batchnorm_7/batchnorm/mul_2Mul3DCNN/batchnorm_7/batchnorm/ReadVariableOp_1:value:0"DCNN/batchnorm_7/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2"
 DCNN/batchnorm_7/batchnorm/mul_2?
+DCNN/batchnorm_7/batchnorm/ReadVariableOp_2ReadVariableOp4dcnn_batchnorm_7_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02-
+DCNN/batchnorm_7/batchnorm/ReadVariableOp_2?
DCNN/batchnorm_7/batchnorm/subSub3DCNN/batchnorm_7/batchnorm/ReadVariableOp_2:value:0$DCNN/batchnorm_7/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2 
DCNN/batchnorm_7/batchnorm/sub?
 DCNN/batchnorm_7/batchnorm/add_1AddV2$DCNN/batchnorm_7/batchnorm/mul_1:z:0"DCNN/batchnorm_7/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2"
 DCNN/batchnorm_7/batchnorm/add_1?
DCNN/dropout_4/IdentityIdentity$DCNN/batchnorm_7/batchnorm/add_1:z:0*
T0*(
_output_shapes
:??????????2
DCNN/dropout_4/Identity?
$DCNN/out_layer/MatMul/ReadVariableOpReadVariableOp-dcnn_out_layer_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02&
$DCNN/out_layer/MatMul/ReadVariableOp?
DCNN/out_layer/MatMulMatMul DCNN/dropout_4/Identity:output:0,DCNN/out_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
DCNN/out_layer/MatMul?
%DCNN/out_layer/BiasAdd/ReadVariableOpReadVariableOp.dcnn_out_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%DCNN/out_layer/BiasAdd/ReadVariableOp?
DCNN/out_layer/BiasAddBiasAddDCNN/out_layer/MatMul:product:0-DCNN/out_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
DCNN/out_layer/BiasAdd?
DCNN/out_layer/SoftmaxSoftmaxDCNN/out_layer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
DCNN/out_layer/Softmax?
IdentityIdentity DCNN/out_layer/Softmax:softmax:01^DCNN/batchnorm_1/FusedBatchNormV3/ReadVariableOp3^DCNN/batchnorm_1/FusedBatchNormV3/ReadVariableOp_1 ^DCNN/batchnorm_1/ReadVariableOp"^DCNN/batchnorm_1/ReadVariableOp_11^DCNN/batchnorm_2/FusedBatchNormV3/ReadVariableOp3^DCNN/batchnorm_2/FusedBatchNormV3/ReadVariableOp_1 ^DCNN/batchnorm_2/ReadVariableOp"^DCNN/batchnorm_2/ReadVariableOp_11^DCNN/batchnorm_3/FusedBatchNormV3/ReadVariableOp3^DCNN/batchnorm_3/FusedBatchNormV3/ReadVariableOp_1 ^DCNN/batchnorm_3/ReadVariableOp"^DCNN/batchnorm_3/ReadVariableOp_11^DCNN/batchnorm_4/FusedBatchNormV3/ReadVariableOp3^DCNN/batchnorm_4/FusedBatchNormV3/ReadVariableOp_1 ^DCNN/batchnorm_4/ReadVariableOp"^DCNN/batchnorm_4/ReadVariableOp_11^DCNN/batchnorm_5/FusedBatchNormV3/ReadVariableOp3^DCNN/batchnorm_5/FusedBatchNormV3/ReadVariableOp_1 ^DCNN/batchnorm_5/ReadVariableOp"^DCNN/batchnorm_5/ReadVariableOp_11^DCNN/batchnorm_6/FusedBatchNormV3/ReadVariableOp3^DCNN/batchnorm_6/FusedBatchNormV3/ReadVariableOp_1 ^DCNN/batchnorm_6/ReadVariableOp"^DCNN/batchnorm_6/ReadVariableOp_1*^DCNN/batchnorm_7/batchnorm/ReadVariableOp,^DCNN/batchnorm_7/batchnorm/ReadVariableOp_1,^DCNN/batchnorm_7/batchnorm/ReadVariableOp_2.^DCNN/batchnorm_7/batchnorm/mul/ReadVariableOp%^DCNN/conv2d_1/BiasAdd/ReadVariableOp$^DCNN/conv2d_1/Conv2D/ReadVariableOp%^DCNN/conv2d_2/BiasAdd/ReadVariableOp$^DCNN/conv2d_2/Conv2D/ReadVariableOp%^DCNN/conv2d_3/BiasAdd/ReadVariableOp$^DCNN/conv2d_3/Conv2D/ReadVariableOp%^DCNN/conv2d_4/BiasAdd/ReadVariableOp$^DCNN/conv2d_4/Conv2D/ReadVariableOp%^DCNN/conv2d_5/BiasAdd/ReadVariableOp$^DCNN/conv2d_5/Conv2D/ReadVariableOp%^DCNN/conv2d_6/BiasAdd/ReadVariableOp$^DCNN/conv2d_6/Conv2D/ReadVariableOp$^DCNN/dense_1/BiasAdd/ReadVariableOp#^DCNN/dense_1/MatMul/ReadVariableOp&^DCNN/out_layer/BiasAdd/ReadVariableOp%^DCNN/out_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????00::::::::::::::::::::::::::::::::::::::::::::2d
0DCNN/batchnorm_1/FusedBatchNormV3/ReadVariableOp0DCNN/batchnorm_1/FusedBatchNormV3/ReadVariableOp2h
2DCNN/batchnorm_1/FusedBatchNormV3/ReadVariableOp_12DCNN/batchnorm_1/FusedBatchNormV3/ReadVariableOp_12B
DCNN/batchnorm_1/ReadVariableOpDCNN/batchnorm_1/ReadVariableOp2F
!DCNN/batchnorm_1/ReadVariableOp_1!DCNN/batchnorm_1/ReadVariableOp_12d
0DCNN/batchnorm_2/FusedBatchNormV3/ReadVariableOp0DCNN/batchnorm_2/FusedBatchNormV3/ReadVariableOp2h
2DCNN/batchnorm_2/FusedBatchNormV3/ReadVariableOp_12DCNN/batchnorm_2/FusedBatchNormV3/ReadVariableOp_12B
DCNN/batchnorm_2/ReadVariableOpDCNN/batchnorm_2/ReadVariableOp2F
!DCNN/batchnorm_2/ReadVariableOp_1!DCNN/batchnorm_2/ReadVariableOp_12d
0DCNN/batchnorm_3/FusedBatchNormV3/ReadVariableOp0DCNN/batchnorm_3/FusedBatchNormV3/ReadVariableOp2h
2DCNN/batchnorm_3/FusedBatchNormV3/ReadVariableOp_12DCNN/batchnorm_3/FusedBatchNormV3/ReadVariableOp_12B
DCNN/batchnorm_3/ReadVariableOpDCNN/batchnorm_3/ReadVariableOp2F
!DCNN/batchnorm_3/ReadVariableOp_1!DCNN/batchnorm_3/ReadVariableOp_12d
0DCNN/batchnorm_4/FusedBatchNormV3/ReadVariableOp0DCNN/batchnorm_4/FusedBatchNormV3/ReadVariableOp2h
2DCNN/batchnorm_4/FusedBatchNormV3/ReadVariableOp_12DCNN/batchnorm_4/FusedBatchNormV3/ReadVariableOp_12B
DCNN/batchnorm_4/ReadVariableOpDCNN/batchnorm_4/ReadVariableOp2F
!DCNN/batchnorm_4/ReadVariableOp_1!DCNN/batchnorm_4/ReadVariableOp_12d
0DCNN/batchnorm_5/FusedBatchNormV3/ReadVariableOp0DCNN/batchnorm_5/FusedBatchNormV3/ReadVariableOp2h
2DCNN/batchnorm_5/FusedBatchNormV3/ReadVariableOp_12DCNN/batchnorm_5/FusedBatchNormV3/ReadVariableOp_12B
DCNN/batchnorm_5/ReadVariableOpDCNN/batchnorm_5/ReadVariableOp2F
!DCNN/batchnorm_5/ReadVariableOp_1!DCNN/batchnorm_5/ReadVariableOp_12d
0DCNN/batchnorm_6/FusedBatchNormV3/ReadVariableOp0DCNN/batchnorm_6/FusedBatchNormV3/ReadVariableOp2h
2DCNN/batchnorm_6/FusedBatchNormV3/ReadVariableOp_12DCNN/batchnorm_6/FusedBatchNormV3/ReadVariableOp_12B
DCNN/batchnorm_6/ReadVariableOpDCNN/batchnorm_6/ReadVariableOp2F
!DCNN/batchnorm_6/ReadVariableOp_1!DCNN/batchnorm_6/ReadVariableOp_12V
)DCNN/batchnorm_7/batchnorm/ReadVariableOp)DCNN/batchnorm_7/batchnorm/ReadVariableOp2Z
+DCNN/batchnorm_7/batchnorm/ReadVariableOp_1+DCNN/batchnorm_7/batchnorm/ReadVariableOp_12Z
+DCNN/batchnorm_7/batchnorm/ReadVariableOp_2+DCNN/batchnorm_7/batchnorm/ReadVariableOp_22^
-DCNN/batchnorm_7/batchnorm/mul/ReadVariableOp-DCNN/batchnorm_7/batchnorm/mul/ReadVariableOp2L
$DCNN/conv2d_1/BiasAdd/ReadVariableOp$DCNN/conv2d_1/BiasAdd/ReadVariableOp2J
#DCNN/conv2d_1/Conv2D/ReadVariableOp#DCNN/conv2d_1/Conv2D/ReadVariableOp2L
$DCNN/conv2d_2/BiasAdd/ReadVariableOp$DCNN/conv2d_2/BiasAdd/ReadVariableOp2J
#DCNN/conv2d_2/Conv2D/ReadVariableOp#DCNN/conv2d_2/Conv2D/ReadVariableOp2L
$DCNN/conv2d_3/BiasAdd/ReadVariableOp$DCNN/conv2d_3/BiasAdd/ReadVariableOp2J
#DCNN/conv2d_3/Conv2D/ReadVariableOp#DCNN/conv2d_3/Conv2D/ReadVariableOp2L
$DCNN/conv2d_4/BiasAdd/ReadVariableOp$DCNN/conv2d_4/BiasAdd/ReadVariableOp2J
#DCNN/conv2d_4/Conv2D/ReadVariableOp#DCNN/conv2d_4/Conv2D/ReadVariableOp2L
$DCNN/conv2d_5/BiasAdd/ReadVariableOp$DCNN/conv2d_5/BiasAdd/ReadVariableOp2J
#DCNN/conv2d_5/Conv2D/ReadVariableOp#DCNN/conv2d_5/Conv2D/ReadVariableOp2L
$DCNN/conv2d_6/BiasAdd/ReadVariableOp$DCNN/conv2d_6/BiasAdd/ReadVariableOp2J
#DCNN/conv2d_6/Conv2D/ReadVariableOp#DCNN/conv2d_6/Conv2D/ReadVariableOp2J
#DCNN/dense_1/BiasAdd/ReadVariableOp#DCNN/dense_1/BiasAdd/ReadVariableOp2H
"DCNN/dense_1/MatMul/ReadVariableOp"DCNN/dense_1/MatMul/ReadVariableOp2N
%DCNN/out_layer/BiasAdd/ReadVariableOp%DCNN/out_layer/BiasAdd/ReadVariableOp2L
$DCNN/out_layer/MatMul/ReadVariableOp$DCNN/out_layer/MatMul/ReadVariableOp:. *
(
_user_specified_nameconv2d_1_input
?
?
C__inference_conv2d_5_layer_call_and_return_conditional_losses_25722

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAddp
EluEluBiasAdd:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
Elu?
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
?#
?
F__inference_batchnorm_1_layer_call_and_return_conditional_losses_28142

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_28127
assignmovingavg_1_28134
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:?????????00@:@:@:@:@:*
epsilon%o?:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *?p}?2	
Const_2?
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/28127*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/28127*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_28127*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/28127*
_output_shapes
:@2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/28127*
_output_shapes
:@2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_28127AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/28127*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/28134*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/28134*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_28134*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/28134*
_output_shapes
:@2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/28134*
_output_shapes
:@2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_28134AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/28134*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????00@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????00@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?
?
+__inference_batchnorm_4_layer_call_fn_28688

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_4_layer_call_and_return_conditional_losses_264962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?$
?
F__inference_batchnorm_5_layer_call_and_return_conditional_losses_28852

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_28837
assignmovingavg_1_28844
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *?p}?2	
Const_2?
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/28837*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/28837*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_28837*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/28837*
_output_shapes	
:?2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/28837*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_28837AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/28837*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/28844*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/28844*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_28844*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/28844*
_output_shapes	
:?2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/28844*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_28844AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/28844*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?
?
(__inference_conv2d_1_layer_call_fn_25094

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_250862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
+__inference_batchnorm_3_layer_call_fn_28454

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_3_layer_call_and_return_conditional_losses_255062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_25239

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAddo
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Elu?
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
F__inference_batchnorm_1_layer_call_and_return_conditional_losses_28164

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
Const?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????00@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????00@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?
?
F__inference_batchnorm_3_layer_call_and_return_conditional_losses_28445

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
Const?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?
?
'__inference_dense_1_layer_call_fn_29116

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_268092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????H::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?$
?
F__inference_batchnorm_6_layer_call_and_return_conditional_losses_25977

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_25962
assignmovingavg_1_25969
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *?p}?2	
Const_2?
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/25962*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/25962*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_25962*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/25962*
_output_shapes	
:?2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/25962*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_25962AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/25962*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/25969*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/25969*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_25969*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/25969*
_output_shapes	
:?2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/25969*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_25969AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/25969*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?#
?
F__inference_batchnorm_3_layer_call_and_return_conditional_losses_28497

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_28482
assignmovingavg_1_28489
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *?p}?2	
Const_2?
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/28482*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/28482*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_28482*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/28482*
_output_shapes	
:?2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/28482*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_28482AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/28482*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/28489*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/28489*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_28489*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/28489*
_output_shapes	
:?2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/28489*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_28489AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/28489*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?
?
+__inference_batchnorm_3_layer_call_fn_28463

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_3_layer_call_and_return_conditional_losses_255372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?$
?
F__inference_batchnorm_2_layer_call_and_return_conditional_losses_25341

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_25326
assignmovingavg_1_25333
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *?p}?2	
Const_2?
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/25326*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/25326*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_25326*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/25326*
_output_shapes
:@2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/25326*
_output_shapes
:@2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_25326AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/25326*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/25333*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/25333*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_25333*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/25333*
_output_shapes
:@2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/25333*
_output_shapes
:@2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_25333AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/25333*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_28722

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
?
+__inference_batchnorm_7_layer_call_fn_29232

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_7_layer_call_and_return_conditional_losses_261642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
+__inference_batchnorm_6_layer_call_fn_28978

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_6_layer_call_and_return_conditional_losses_260082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
c
D__inference_dropout_4_layer_call_and_return_conditional_losses_26864

inputs
identity?a
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *???2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/random_uniform/max?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniform?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:??????????2
dropout/random_uniform/mul?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:??????????2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqualq
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:??????????2
dropout/mul?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/mul_1f
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
?
$__inference_DCNN_layer_call_fn_27218
conv2d_1_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32#
statefulpartitionedcall_args_33#
statefulpartitionedcall_args_34#
statefulpartitionedcall_args_35#
statefulpartitionedcall_args_36#
statefulpartitionedcall_args_37#
statefulpartitionedcall_args_38#
statefulpartitionedcall_args_39#
statefulpartitionedcall_args_40#
statefulpartitionedcall_args_41#
statefulpartitionedcall_args_42#
statefulpartitionedcall_args_43#
statefulpartitionedcall_args_44
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32statefulpartitionedcall_args_33statefulpartitionedcall_args_34statefulpartitionedcall_args_35statefulpartitionedcall_args_36statefulpartitionedcall_args_37statefulpartitionedcall_args_38statefulpartitionedcall_args_39statefulpartitionedcall_args_40statefulpartitionedcall_args_41statefulpartitionedcall_args_42statefulpartitionedcall_args_43statefulpartitionedcall_args_44*8
Tin1
/2-*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_DCNN_layer_call_and_return_conditional_losses_271712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????00::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:. *
(
_user_specified_nameconv2d_1_input
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_28367

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????@:& "
 
_user_specified_nameinputs
?
E
)__inference_dropout_2_layer_call_fn_28732

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_265682
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_26790

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? $  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????H2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????H2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
?
F__inference_batchnorm_5_layer_call_and_return_conditional_losses_26639

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
Const?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?	
?
D__inference_out_layer_layer_call_and_return_conditional_losses_26893

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?$
?
F__inference_batchnorm_1_layer_call_and_return_conditional_losses_28068

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_28053
assignmovingavg_1_28060
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *?p}?2	
Const_2?
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/28053*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/28053*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_28053*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/28053*
_output_shapes
:@2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/28053*
_output_shapes
:@2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_28053AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/28053*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/28060*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/28060*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_28060*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/28060*
_output_shapes
:@2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/28060*
_output_shapes
:@2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_28060AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/28060*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?
E
)__inference_dropout_1_layer_call_fn_28377

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????@**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_263652
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:& "
 
_user_specified_nameinputs
?	
?
B__inference_dense_1_layer_call_and_return_conditional_losses_29109

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?H?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddV
EluEluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Elu?
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????H::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
+__inference_batchnorm_5_layer_call_fn_28883

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_5_layer_call_and_return_conditional_losses_258242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?	
?
D__inference_out_layer_layer_call_and_return_conditional_losses_29278

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?#
?
F__inference_batchnorm_4_layer_call_and_return_conditional_losses_26496

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_26481
assignmovingavg_1_26488
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *?p}?2	
Const_2?
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/26481*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/26481*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_26481*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/26481*
_output_shapes	
:?2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/26481*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_26481AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/26481*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/26488*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/26488*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_26488*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/26488*
_output_shapes	
:?2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/26488*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_26488AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/26488*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?$
?
F__inference_batchnorm_3_layer_call_and_return_conditional_losses_25506

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_25491
assignmovingavg_1_25498
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *?p}?2	
Const_2?
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/25491*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/25491*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_25491*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/25491*
_output_shapes	
:?2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/25491*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_25491AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/25491*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/25498*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/25498*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_25498*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/25498*
_output_shapes	
:?2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/25498*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_25498AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/25498*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?
?
+__inference_batchnorm_4_layer_call_fn_28614

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_4_layer_call_and_return_conditional_losses_256592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
F__inference_batchnorm_4_layer_call_and_return_conditional_losses_26518

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
Const?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_26869

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
?
+__inference_batchnorm_3_layer_call_fn_28528

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_3_layer_call_and_return_conditional_losses_264142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?/
?
F__inference_batchnorm_7_layer_call_and_return_conditional_losses_29191

inputs
assignmovingavg_29166
assignmovingavg_1_29172)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst*(
_class
loc:@AssignMovingAvg/29166*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_29166*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*(
_class
loc:@AssignMovingAvg/29166*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*(
_class
loc:@AssignMovingAvg/29166*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_29166AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/29166*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst**
_class 
loc:@AssignMovingAvg_1/29172*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_29172*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/29172*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/29172*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_29172AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/29172*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_29077

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?#
?
F__inference_batchnorm_4_layer_call_and_return_conditional_losses_28657

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_28642
assignmovingavg_1_28649
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *?p}?2	
Const_2?
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/28642*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/28642*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_28642*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/28642*
_output_shapes	
:?2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/28642*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_28642AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/28642*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/28649*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/28649*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_28649*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/28649*
_output_shapes	
:?2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/28649*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_28649AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/28649*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?
?
+__inference_batchnorm_1_layer_call_fn_28182

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????00@**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_1_layer_call_and_return_conditional_losses_262332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????00@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????00@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
+__inference_batchnorm_7_layer_call_fn_29223

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_7_layer_call_and_return_conditional_losses_261322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
c
D__inference_dropout_3_layer_call_and_return_conditional_losses_29072

inputs
identity?a
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/random_uniform/max?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniform?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*0
_output_shapes
:??????????2
dropout/random_uniform/mul?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*0
_output_shapes
:??????????2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*0
_output_shapes
:??????????2
dropout/GreaterEqualy
dropout/mulMulinputsdropout/truediv:z:0*
T0*0
_output_shapes
:??????????2
dropout/mul?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
dropout/Cast?
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
dropout/mul_1n
IdentityIdentitydropout/mul_1:z:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
?
F__inference_batchnorm_1_layer_call_and_return_conditional_losses_26233

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
Const?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????00@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????00@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?
?
F__inference_batchnorm_6_layer_call_and_return_conditional_losses_28960

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
Const?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_28717

inputs
identity?a
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/random_uniform/max?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniform?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*0
_output_shapes
:??????????2
dropout/random_uniform/mul?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*0
_output_shapes
:??????????2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*0
_output_shapes
:??????????2
dropout/GreaterEqualy
dropout/mulMulinputsdropout/truediv:z:0*
T0*0
_output_shapes
:??????????2
dropout/mul?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
dropout/Cast?
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
dropout/mul_1n
IdentityIdentitydropout/mul_1:z:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
?
)__inference_out_layer_layer_call_fn_29285

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_out_layer_layer_call_and_return_conditional_losses_268932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
C__inference_conv2d_6_layer_call_and_return_conditional_losses_25875

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAddp
EluEluBiasAdd:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
Elu?
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
F__inference_batchnorm_4_layer_call_and_return_conditional_losses_28605

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
Const?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_26771

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
??
?!
?__inference_DCNN_layer_call_and_return_conditional_losses_27728

inputs+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource'
#batchnorm_1_readvariableop_resource)
%batchnorm_1_readvariableop_1_resource%
!batchnorm_1_assignmovingavg_27410'
#batchnorm_1_assignmovingavg_1_27417+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource'
#batchnorm_2_readvariableop_resource)
%batchnorm_2_readvariableop_1_resource%
!batchnorm_2_assignmovingavg_27447'
#batchnorm_2_assignmovingavg_1_27454+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource'
#batchnorm_3_readvariableop_resource)
%batchnorm_3_readvariableop_1_resource%
!batchnorm_3_assignmovingavg_27501'
#batchnorm_3_assignmovingavg_1_27508+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource'
#batchnorm_4_readvariableop_resource)
%batchnorm_4_readvariableop_1_resource%
!batchnorm_4_assignmovingavg_27538'
#batchnorm_4_assignmovingavg_1_27545+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource'
#batchnorm_5_readvariableop_resource)
%batchnorm_5_readvariableop_1_resource%
!batchnorm_5_assignmovingavg_27592'
#batchnorm_5_assignmovingavg_1_27599+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource'
#batchnorm_6_readvariableop_resource)
%batchnorm_6_readvariableop_1_resource%
!batchnorm_6_assignmovingavg_27629'
#batchnorm_6_assignmovingavg_1_27636*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource%
!batchnorm_7_assignmovingavg_27680'
#batchnorm_7_assignmovingavg_1_276865
1batchnorm_7_batchnorm_mul_readvariableop_resource1
-batchnorm_7_batchnorm_readvariableop_resource,
(out_layer_matmul_readvariableop_resource-
)out_layer_biasadd_readvariableop_resource
identity??/batchnorm_1/AssignMovingAvg/AssignSubVariableOp?*batchnorm_1/AssignMovingAvg/ReadVariableOp?1batchnorm_1/AssignMovingAvg_1/AssignSubVariableOp?,batchnorm_1/AssignMovingAvg_1/ReadVariableOp?batchnorm_1/ReadVariableOp?batchnorm_1/ReadVariableOp_1?/batchnorm_2/AssignMovingAvg/AssignSubVariableOp?*batchnorm_2/AssignMovingAvg/ReadVariableOp?1batchnorm_2/AssignMovingAvg_1/AssignSubVariableOp?,batchnorm_2/AssignMovingAvg_1/ReadVariableOp?batchnorm_2/ReadVariableOp?batchnorm_2/ReadVariableOp_1?/batchnorm_3/AssignMovingAvg/AssignSubVariableOp?*batchnorm_3/AssignMovingAvg/ReadVariableOp?1batchnorm_3/AssignMovingAvg_1/AssignSubVariableOp?,batchnorm_3/AssignMovingAvg_1/ReadVariableOp?batchnorm_3/ReadVariableOp?batchnorm_3/ReadVariableOp_1?/batchnorm_4/AssignMovingAvg/AssignSubVariableOp?*batchnorm_4/AssignMovingAvg/ReadVariableOp?1batchnorm_4/AssignMovingAvg_1/AssignSubVariableOp?,batchnorm_4/AssignMovingAvg_1/ReadVariableOp?batchnorm_4/ReadVariableOp?batchnorm_4/ReadVariableOp_1?/batchnorm_5/AssignMovingAvg/AssignSubVariableOp?*batchnorm_5/AssignMovingAvg/ReadVariableOp?1batchnorm_5/AssignMovingAvg_1/AssignSubVariableOp?,batchnorm_5/AssignMovingAvg_1/ReadVariableOp?batchnorm_5/ReadVariableOp?batchnorm_5/ReadVariableOp_1?/batchnorm_6/AssignMovingAvg/AssignSubVariableOp?*batchnorm_6/AssignMovingAvg/ReadVariableOp?1batchnorm_6/AssignMovingAvg_1/AssignSubVariableOp?,batchnorm_6/AssignMovingAvg_1/ReadVariableOp?batchnorm_6/ReadVariableOp?batchnorm_6/ReadVariableOp_1?/batchnorm_7/AssignMovingAvg/AssignSubVariableOp?*batchnorm_7/AssignMovingAvg/ReadVariableOp?1batchnorm_7/AssignMovingAvg_1/AssignSubVariableOp?,batchnorm_7/AssignMovingAvg_1/ReadVariableOp?$batchnorm_7/batchnorm/ReadVariableOp?(batchnorm_7/batchnorm/mul/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp? out_layer/BiasAdd/ReadVariableOp?out_layer/MatMul/ReadVariableOp?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00@*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00@2
conv2d_1/BiasAddx
conv2d_1/EluEluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00@2
conv2d_1/Eluv
batchnorm_1/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
batchnorm_1/LogicalAnd/xv
batchnorm_1/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
batchnorm_1/LogicalAnd/y?
batchnorm_1/LogicalAnd
LogicalAnd!batchnorm_1/LogicalAnd/x:output:0!batchnorm_1/LogicalAnd/y:output:0*
_output_shapes
: 2
batchnorm_1/LogicalAnd?
batchnorm_1/ReadVariableOpReadVariableOp#batchnorm_1_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm_1/ReadVariableOp?
batchnorm_1/ReadVariableOp_1ReadVariableOp%batchnorm_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm_1/ReadVariableOp_1i
batchnorm_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
batchnorm_1/Constm
batchnorm_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
batchnorm_1/Const_1?
batchnorm_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/Elu:activations:0"batchnorm_1/ReadVariableOp:value:0$batchnorm_1/ReadVariableOp_1:value:0batchnorm_1/Const:output:0batchnorm_1/Const_1:output:0*
T0*
U0*K
_output_shapes9
7:?????????00@:@:@:@:@:*
epsilon%o?:2
batchnorm_1/FusedBatchNormV3o
batchnorm_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
batchnorm_1/Const_2?
!batchnorm_1/AssignMovingAvg/sub/xConst*4
_class*
(&loc:@batchnorm_1/AssignMovingAvg/27410*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!batchnorm_1/AssignMovingAvg/sub/x?
batchnorm_1/AssignMovingAvg/subSub*batchnorm_1/AssignMovingAvg/sub/x:output:0batchnorm_1/Const_2:output:0*
T0*4
_class*
(&loc:@batchnorm_1/AssignMovingAvg/27410*
_output_shapes
: 2!
batchnorm_1/AssignMovingAvg/sub?
*batchnorm_1/AssignMovingAvg/ReadVariableOpReadVariableOp!batchnorm_1_assignmovingavg_27410*
_output_shapes
:@*
dtype02,
*batchnorm_1/AssignMovingAvg/ReadVariableOp?
!batchnorm_1/AssignMovingAvg/sub_1Sub2batchnorm_1/AssignMovingAvg/ReadVariableOp:value:0)batchnorm_1/FusedBatchNormV3:batch_mean:0*
T0*4
_class*
(&loc:@batchnorm_1/AssignMovingAvg/27410*
_output_shapes
:@2#
!batchnorm_1/AssignMovingAvg/sub_1?
batchnorm_1/AssignMovingAvg/mulMul%batchnorm_1/AssignMovingAvg/sub_1:z:0#batchnorm_1/AssignMovingAvg/sub:z:0*
T0*4
_class*
(&loc:@batchnorm_1/AssignMovingAvg/27410*
_output_shapes
:@2!
batchnorm_1/AssignMovingAvg/mul?
/batchnorm_1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batchnorm_1_assignmovingavg_27410#batchnorm_1/AssignMovingAvg/mul:z:0+^batchnorm_1/AssignMovingAvg/ReadVariableOp*4
_class*
(&loc:@batchnorm_1/AssignMovingAvg/27410*
_output_shapes
 *
dtype021
/batchnorm_1/AssignMovingAvg/AssignSubVariableOp?
#batchnorm_1/AssignMovingAvg_1/sub/xConst*6
_class,
*(loc:@batchnorm_1/AssignMovingAvg_1/27417*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#batchnorm_1/AssignMovingAvg_1/sub/x?
!batchnorm_1/AssignMovingAvg_1/subSub,batchnorm_1/AssignMovingAvg_1/sub/x:output:0batchnorm_1/Const_2:output:0*
T0*6
_class,
*(loc:@batchnorm_1/AssignMovingAvg_1/27417*
_output_shapes
: 2#
!batchnorm_1/AssignMovingAvg_1/sub?
,batchnorm_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp#batchnorm_1_assignmovingavg_1_27417*
_output_shapes
:@*
dtype02.
,batchnorm_1/AssignMovingAvg_1/ReadVariableOp?
#batchnorm_1/AssignMovingAvg_1/sub_1Sub4batchnorm_1/AssignMovingAvg_1/ReadVariableOp:value:0-batchnorm_1/FusedBatchNormV3:batch_variance:0*
T0*6
_class,
*(loc:@batchnorm_1/AssignMovingAvg_1/27417*
_output_shapes
:@2%
#batchnorm_1/AssignMovingAvg_1/sub_1?
!batchnorm_1/AssignMovingAvg_1/mulMul'batchnorm_1/AssignMovingAvg_1/sub_1:z:0%batchnorm_1/AssignMovingAvg_1/sub:z:0*
T0*6
_class,
*(loc:@batchnorm_1/AssignMovingAvg_1/27417*
_output_shapes
:@2#
!batchnorm_1/AssignMovingAvg_1/mul?
1batchnorm_1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp#batchnorm_1_assignmovingavg_1_27417%batchnorm_1/AssignMovingAvg_1/mul:z:0-^batchnorm_1/AssignMovingAvg_1/ReadVariableOp*6
_class,
*(loc:@batchnorm_1/AssignMovingAvg_1/27417*
_output_shapes
 *
dtype023
1batchnorm_1/AssignMovingAvg_1/AssignSubVariableOp?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2D batchnorm_1/FusedBatchNormV3:y:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00@*
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00@2
conv2d_2/BiasAddx
conv2d_2/EluEluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00@2
conv2d_2/Eluv
batchnorm_2/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
batchnorm_2/LogicalAnd/xv
batchnorm_2/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
batchnorm_2/LogicalAnd/y?
batchnorm_2/LogicalAnd
LogicalAnd!batchnorm_2/LogicalAnd/x:output:0!batchnorm_2/LogicalAnd/y:output:0*
_output_shapes
: 2
batchnorm_2/LogicalAnd?
batchnorm_2/ReadVariableOpReadVariableOp#batchnorm_2_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm_2/ReadVariableOp?
batchnorm_2/ReadVariableOp_1ReadVariableOp%batchnorm_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm_2/ReadVariableOp_1i
batchnorm_2/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
batchnorm_2/Constm
batchnorm_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
batchnorm_2/Const_1?
batchnorm_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/Elu:activations:0"batchnorm_2/ReadVariableOp:value:0$batchnorm_2/ReadVariableOp_1:value:0batchnorm_2/Const:output:0batchnorm_2/Const_1:output:0*
T0*
U0*K
_output_shapes9
7:?????????00@:@:@:@:@:*
epsilon%o?:2
batchnorm_2/FusedBatchNormV3o
batchnorm_2/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
batchnorm_2/Const_2?
!batchnorm_2/AssignMovingAvg/sub/xConst*4
_class*
(&loc:@batchnorm_2/AssignMovingAvg/27447*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!batchnorm_2/AssignMovingAvg/sub/x?
batchnorm_2/AssignMovingAvg/subSub*batchnorm_2/AssignMovingAvg/sub/x:output:0batchnorm_2/Const_2:output:0*
T0*4
_class*
(&loc:@batchnorm_2/AssignMovingAvg/27447*
_output_shapes
: 2!
batchnorm_2/AssignMovingAvg/sub?
*batchnorm_2/AssignMovingAvg/ReadVariableOpReadVariableOp!batchnorm_2_assignmovingavg_27447*
_output_shapes
:@*
dtype02,
*batchnorm_2/AssignMovingAvg/ReadVariableOp?
!batchnorm_2/AssignMovingAvg/sub_1Sub2batchnorm_2/AssignMovingAvg/ReadVariableOp:value:0)batchnorm_2/FusedBatchNormV3:batch_mean:0*
T0*4
_class*
(&loc:@batchnorm_2/AssignMovingAvg/27447*
_output_shapes
:@2#
!batchnorm_2/AssignMovingAvg/sub_1?
batchnorm_2/AssignMovingAvg/mulMul%batchnorm_2/AssignMovingAvg/sub_1:z:0#batchnorm_2/AssignMovingAvg/sub:z:0*
T0*4
_class*
(&loc:@batchnorm_2/AssignMovingAvg/27447*
_output_shapes
:@2!
batchnorm_2/AssignMovingAvg/mul?
/batchnorm_2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batchnorm_2_assignmovingavg_27447#batchnorm_2/AssignMovingAvg/mul:z:0+^batchnorm_2/AssignMovingAvg/ReadVariableOp*4
_class*
(&loc:@batchnorm_2/AssignMovingAvg/27447*
_output_shapes
 *
dtype021
/batchnorm_2/AssignMovingAvg/AssignSubVariableOp?
#batchnorm_2/AssignMovingAvg_1/sub/xConst*6
_class,
*(loc:@batchnorm_2/AssignMovingAvg_1/27454*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#batchnorm_2/AssignMovingAvg_1/sub/x?
!batchnorm_2/AssignMovingAvg_1/subSub,batchnorm_2/AssignMovingAvg_1/sub/x:output:0batchnorm_2/Const_2:output:0*
T0*6
_class,
*(loc:@batchnorm_2/AssignMovingAvg_1/27454*
_output_shapes
: 2#
!batchnorm_2/AssignMovingAvg_1/sub?
,batchnorm_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp#batchnorm_2_assignmovingavg_1_27454*
_output_shapes
:@*
dtype02.
,batchnorm_2/AssignMovingAvg_1/ReadVariableOp?
#batchnorm_2/AssignMovingAvg_1/sub_1Sub4batchnorm_2/AssignMovingAvg_1/ReadVariableOp:value:0-batchnorm_2/FusedBatchNormV3:batch_variance:0*
T0*6
_class,
*(loc:@batchnorm_2/AssignMovingAvg_1/27454*
_output_shapes
:@2%
#batchnorm_2/AssignMovingAvg_1/sub_1?
!batchnorm_2/AssignMovingAvg_1/mulMul'batchnorm_2/AssignMovingAvg_1/sub_1:z:0%batchnorm_2/AssignMovingAvg_1/sub:z:0*
T0*6
_class,
*(loc:@batchnorm_2/AssignMovingAvg_1/27454*
_output_shapes
:@2#
!batchnorm_2/AssignMovingAvg_1/mul?
1batchnorm_2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp#batchnorm_2_assignmovingavg_1_27454%batchnorm_2/AssignMovingAvg_1/mul:z:0-^batchnorm_2/AssignMovingAvg_1/ReadVariableOp*6
_class,
*(loc:@batchnorm_2/AssignMovingAvg_1/27454*
_output_shapes
 *
dtype023
1batchnorm_2/AssignMovingAvg_1/AssignSubVariableOp?
maxpool2d_1/MaxPoolMaxPool batchnorm_2/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
maxpool2d_1/MaxPoolu
dropout_1/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout_1/dropout/rate~
dropout_1/dropout/ShapeShapemaxpool2d_1/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape?
$dropout_1/dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$dropout_1/dropout/random_uniform/min?
$dropout_1/dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$dropout_1/dropout/random_uniform/max?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform?
$dropout_1/dropout/random_uniform/subSub-dropout_1/dropout/random_uniform/max:output:0-dropout_1/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2&
$dropout_1/dropout/random_uniform/sub?
$dropout_1/dropout/random_uniform/mulMul7dropout_1/dropout/random_uniform/RandomUniform:output:0(dropout_1/dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:?????????@2&
$dropout_1/dropout/random_uniform/mul?
 dropout_1/dropout/random_uniformAdd(dropout_1/dropout/random_uniform/mul:z:0-dropout_1/dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:?????????@2"
 dropout_1/dropout/random_uniformw
dropout_1/dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_1/dropout/sub/x?
dropout_1/dropout/subSub dropout_1/dropout/sub/x:output:0dropout_1/dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout_1/dropout/sub
dropout_1/dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_1/dropout/truediv/x?
dropout_1/dropout/truedivRealDiv$dropout_1/dropout/truediv/x:output:0dropout_1/dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout_1/dropout/truediv?
dropout_1/dropout/GreaterEqualGreaterEqual$dropout_1/dropout/random_uniform:z:0dropout_1/dropout/rate:output:0*
T0*/
_output_shapes
:?????????@2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/mulMulmaxpool2d_1/MaxPool:output:0dropout_1/dropout/truediv:z:0*
T0*/
_output_shapes
:?????????@2
dropout_1/dropout/mul?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
dropout_1/dropout/Cast?
dropout_1/dropout/mul_1Muldropout_1/dropout/mul:z:0dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
dropout_1/dropout/mul_1?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Ddropout_1/dropout/mul_1:z:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_3/BiasAddy
conv2d_3/EluEluconv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_3/Eluv
batchnorm_3/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
batchnorm_3/LogicalAnd/xv
batchnorm_3/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
batchnorm_3/LogicalAnd/y?
batchnorm_3/LogicalAnd
LogicalAnd!batchnorm_3/LogicalAnd/x:output:0!batchnorm_3/LogicalAnd/y:output:0*
_output_shapes
: 2
batchnorm_3/LogicalAnd?
batchnorm_3/ReadVariableOpReadVariableOp#batchnorm_3_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm_3/ReadVariableOp?
batchnorm_3/ReadVariableOp_1ReadVariableOp%batchnorm_3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm_3/ReadVariableOp_1i
batchnorm_3/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
batchnorm_3/Constm
batchnorm_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
batchnorm_3/Const_1?
batchnorm_3/FusedBatchNormV3FusedBatchNormV3conv2d_3/Elu:activations:0"batchnorm_3/ReadVariableOp:value:0$batchnorm_3/ReadVariableOp_1:value:0batchnorm_3/Const:output:0batchnorm_3/Const_1:output:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:2
batchnorm_3/FusedBatchNormV3o
batchnorm_3/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
batchnorm_3/Const_2?
!batchnorm_3/AssignMovingAvg/sub/xConst*4
_class*
(&loc:@batchnorm_3/AssignMovingAvg/27501*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!batchnorm_3/AssignMovingAvg/sub/x?
batchnorm_3/AssignMovingAvg/subSub*batchnorm_3/AssignMovingAvg/sub/x:output:0batchnorm_3/Const_2:output:0*
T0*4
_class*
(&loc:@batchnorm_3/AssignMovingAvg/27501*
_output_shapes
: 2!
batchnorm_3/AssignMovingAvg/sub?
*batchnorm_3/AssignMovingAvg/ReadVariableOpReadVariableOp!batchnorm_3_assignmovingavg_27501*
_output_shapes	
:?*
dtype02,
*batchnorm_3/AssignMovingAvg/ReadVariableOp?
!batchnorm_3/AssignMovingAvg/sub_1Sub2batchnorm_3/AssignMovingAvg/ReadVariableOp:value:0)batchnorm_3/FusedBatchNormV3:batch_mean:0*
T0*4
_class*
(&loc:@batchnorm_3/AssignMovingAvg/27501*
_output_shapes	
:?2#
!batchnorm_3/AssignMovingAvg/sub_1?
batchnorm_3/AssignMovingAvg/mulMul%batchnorm_3/AssignMovingAvg/sub_1:z:0#batchnorm_3/AssignMovingAvg/sub:z:0*
T0*4
_class*
(&loc:@batchnorm_3/AssignMovingAvg/27501*
_output_shapes	
:?2!
batchnorm_3/AssignMovingAvg/mul?
/batchnorm_3/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batchnorm_3_assignmovingavg_27501#batchnorm_3/AssignMovingAvg/mul:z:0+^batchnorm_3/AssignMovingAvg/ReadVariableOp*4
_class*
(&loc:@batchnorm_3/AssignMovingAvg/27501*
_output_shapes
 *
dtype021
/batchnorm_3/AssignMovingAvg/AssignSubVariableOp?
#batchnorm_3/AssignMovingAvg_1/sub/xConst*6
_class,
*(loc:@batchnorm_3/AssignMovingAvg_1/27508*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#batchnorm_3/AssignMovingAvg_1/sub/x?
!batchnorm_3/AssignMovingAvg_1/subSub,batchnorm_3/AssignMovingAvg_1/sub/x:output:0batchnorm_3/Const_2:output:0*
T0*6
_class,
*(loc:@batchnorm_3/AssignMovingAvg_1/27508*
_output_shapes
: 2#
!batchnorm_3/AssignMovingAvg_1/sub?
,batchnorm_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp#batchnorm_3_assignmovingavg_1_27508*
_output_shapes	
:?*
dtype02.
,batchnorm_3/AssignMovingAvg_1/ReadVariableOp?
#batchnorm_3/AssignMovingAvg_1/sub_1Sub4batchnorm_3/AssignMovingAvg_1/ReadVariableOp:value:0-batchnorm_3/FusedBatchNormV3:batch_variance:0*
T0*6
_class,
*(loc:@batchnorm_3/AssignMovingAvg_1/27508*
_output_shapes	
:?2%
#batchnorm_3/AssignMovingAvg_1/sub_1?
!batchnorm_3/AssignMovingAvg_1/mulMul'batchnorm_3/AssignMovingAvg_1/sub_1:z:0%batchnorm_3/AssignMovingAvg_1/sub:z:0*
T0*6
_class,
*(loc:@batchnorm_3/AssignMovingAvg_1/27508*
_output_shapes	
:?2#
!batchnorm_3/AssignMovingAvg_1/mul?
1batchnorm_3/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp#batchnorm_3_assignmovingavg_1_27508%batchnorm_3/AssignMovingAvg_1/mul:z:0-^batchnorm_3/AssignMovingAvg_1/ReadVariableOp*6
_class,
*(loc:@batchnorm_3/AssignMovingAvg_1/27508*
_output_shapes
 *
dtype023
1batchnorm_3/AssignMovingAvg_1/AssignSubVariableOp?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2D batchnorm_3/FusedBatchNormV3:y:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_4/BiasAddy
conv2d_4/EluEluconv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_4/Eluv
batchnorm_4/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
batchnorm_4/LogicalAnd/xv
batchnorm_4/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
batchnorm_4/LogicalAnd/y?
batchnorm_4/LogicalAnd
LogicalAnd!batchnorm_4/LogicalAnd/x:output:0!batchnorm_4/LogicalAnd/y:output:0*
_output_shapes
: 2
batchnorm_4/LogicalAnd?
batchnorm_4/ReadVariableOpReadVariableOp#batchnorm_4_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm_4/ReadVariableOp?
batchnorm_4/ReadVariableOp_1ReadVariableOp%batchnorm_4_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm_4/ReadVariableOp_1i
batchnorm_4/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
batchnorm_4/Constm
batchnorm_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
batchnorm_4/Const_1?
batchnorm_4/FusedBatchNormV3FusedBatchNormV3conv2d_4/Elu:activations:0"batchnorm_4/ReadVariableOp:value:0$batchnorm_4/ReadVariableOp_1:value:0batchnorm_4/Const:output:0batchnorm_4/Const_1:output:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:2
batchnorm_4/FusedBatchNormV3o
batchnorm_4/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
batchnorm_4/Const_2?
!batchnorm_4/AssignMovingAvg/sub/xConst*4
_class*
(&loc:@batchnorm_4/AssignMovingAvg/27538*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!batchnorm_4/AssignMovingAvg/sub/x?
batchnorm_4/AssignMovingAvg/subSub*batchnorm_4/AssignMovingAvg/sub/x:output:0batchnorm_4/Const_2:output:0*
T0*4
_class*
(&loc:@batchnorm_4/AssignMovingAvg/27538*
_output_shapes
: 2!
batchnorm_4/AssignMovingAvg/sub?
*batchnorm_4/AssignMovingAvg/ReadVariableOpReadVariableOp!batchnorm_4_assignmovingavg_27538*
_output_shapes	
:?*
dtype02,
*batchnorm_4/AssignMovingAvg/ReadVariableOp?
!batchnorm_4/AssignMovingAvg/sub_1Sub2batchnorm_4/AssignMovingAvg/ReadVariableOp:value:0)batchnorm_4/FusedBatchNormV3:batch_mean:0*
T0*4
_class*
(&loc:@batchnorm_4/AssignMovingAvg/27538*
_output_shapes	
:?2#
!batchnorm_4/AssignMovingAvg/sub_1?
batchnorm_4/AssignMovingAvg/mulMul%batchnorm_4/AssignMovingAvg/sub_1:z:0#batchnorm_4/AssignMovingAvg/sub:z:0*
T0*4
_class*
(&loc:@batchnorm_4/AssignMovingAvg/27538*
_output_shapes	
:?2!
batchnorm_4/AssignMovingAvg/mul?
/batchnorm_4/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batchnorm_4_assignmovingavg_27538#batchnorm_4/AssignMovingAvg/mul:z:0+^batchnorm_4/AssignMovingAvg/ReadVariableOp*4
_class*
(&loc:@batchnorm_4/AssignMovingAvg/27538*
_output_shapes
 *
dtype021
/batchnorm_4/AssignMovingAvg/AssignSubVariableOp?
#batchnorm_4/AssignMovingAvg_1/sub/xConst*6
_class,
*(loc:@batchnorm_4/AssignMovingAvg_1/27545*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#batchnorm_4/AssignMovingAvg_1/sub/x?
!batchnorm_4/AssignMovingAvg_1/subSub,batchnorm_4/AssignMovingAvg_1/sub/x:output:0batchnorm_4/Const_2:output:0*
T0*6
_class,
*(loc:@batchnorm_4/AssignMovingAvg_1/27545*
_output_shapes
: 2#
!batchnorm_4/AssignMovingAvg_1/sub?
,batchnorm_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp#batchnorm_4_assignmovingavg_1_27545*
_output_shapes	
:?*
dtype02.
,batchnorm_4/AssignMovingAvg_1/ReadVariableOp?
#batchnorm_4/AssignMovingAvg_1/sub_1Sub4batchnorm_4/AssignMovingAvg_1/ReadVariableOp:value:0-batchnorm_4/FusedBatchNormV3:batch_variance:0*
T0*6
_class,
*(loc:@batchnorm_4/AssignMovingAvg_1/27545*
_output_shapes	
:?2%
#batchnorm_4/AssignMovingAvg_1/sub_1?
!batchnorm_4/AssignMovingAvg_1/mulMul'batchnorm_4/AssignMovingAvg_1/sub_1:z:0%batchnorm_4/AssignMovingAvg_1/sub:z:0*
T0*6
_class,
*(loc:@batchnorm_4/AssignMovingAvg_1/27545*
_output_shapes	
:?2#
!batchnorm_4/AssignMovingAvg_1/mul?
1batchnorm_4/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp#batchnorm_4_assignmovingavg_1_27545%batchnorm_4/AssignMovingAvg_1/mul:z:0-^batchnorm_4/AssignMovingAvg_1/ReadVariableOp*6
_class,
*(loc:@batchnorm_4/AssignMovingAvg_1/27545*
_output_shapes
 *
dtype023
1batchnorm_4/AssignMovingAvg_1/AssignSubVariableOp?
maxpool2d_2/MaxPoolMaxPool batchnorm_4/FusedBatchNormV3:y:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
maxpool2d_2/MaxPoolu
dropout_2/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout_2/dropout/rate~
dropout_2/dropout/ShapeShapemaxpool2d_2/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shape?
$dropout_2/dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$dropout_2/dropout/random_uniform/min?
$dropout_2/dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$dropout_2/dropout/random_uniform/max?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype020
.dropout_2/dropout/random_uniform/RandomUniform?
$dropout_2/dropout/random_uniform/subSub-dropout_2/dropout/random_uniform/max:output:0-dropout_2/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2&
$dropout_2/dropout/random_uniform/sub?
$dropout_2/dropout/random_uniform/mulMul7dropout_2/dropout/random_uniform/RandomUniform:output:0(dropout_2/dropout/random_uniform/sub:z:0*
T0*0
_output_shapes
:??????????2&
$dropout_2/dropout/random_uniform/mul?
 dropout_2/dropout/random_uniformAdd(dropout_2/dropout/random_uniform/mul:z:0-dropout_2/dropout/random_uniform/min:output:0*
T0*0
_output_shapes
:??????????2"
 dropout_2/dropout/random_uniformw
dropout_2/dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_2/dropout/sub/x?
dropout_2/dropout/subSub dropout_2/dropout/sub/x:output:0dropout_2/dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout_2/dropout/sub
dropout_2/dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_2/dropout/truediv/x?
dropout_2/dropout/truedivRealDiv$dropout_2/dropout/truediv/x:output:0dropout_2/dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout_2/dropout/truediv?
dropout_2/dropout/GreaterEqualGreaterEqual$dropout_2/dropout/random_uniform:z:0dropout_2/dropout/rate:output:0*
T0*0
_output_shapes
:??????????2 
dropout_2/dropout/GreaterEqual?
dropout_2/dropout/mulMulmaxpool2d_2/MaxPool:output:0dropout_2/dropout/truediv:z:0*
T0*0
_output_shapes
:??????????2
dropout_2/dropout/mul?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
dropout_2/dropout/Cast?
dropout_2/dropout/mul_1Muldropout_2/dropout/mul:z:0dropout_2/dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
dropout_2/dropout/mul_1?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2Ddropout_2/dropout/mul_1:z:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_5/BiasAddy
conv2d_5/EluEluconv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_5/Eluv
batchnorm_5/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
batchnorm_5/LogicalAnd/xv
batchnorm_5/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
batchnorm_5/LogicalAnd/y?
batchnorm_5/LogicalAnd
LogicalAnd!batchnorm_5/LogicalAnd/x:output:0!batchnorm_5/LogicalAnd/y:output:0*
_output_shapes
: 2
batchnorm_5/LogicalAnd?
batchnorm_5/ReadVariableOpReadVariableOp#batchnorm_5_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm_5/ReadVariableOp?
batchnorm_5/ReadVariableOp_1ReadVariableOp%batchnorm_5_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm_5/ReadVariableOp_1i
batchnorm_5/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
batchnorm_5/Constm
batchnorm_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
batchnorm_5/Const_1?
batchnorm_5/FusedBatchNormV3FusedBatchNormV3conv2d_5/Elu:activations:0"batchnorm_5/ReadVariableOp:value:0$batchnorm_5/ReadVariableOp_1:value:0batchnorm_5/Const:output:0batchnorm_5/Const_1:output:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:2
batchnorm_5/FusedBatchNormV3o
batchnorm_5/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
batchnorm_5/Const_2?
!batchnorm_5/AssignMovingAvg/sub/xConst*4
_class*
(&loc:@batchnorm_5/AssignMovingAvg/27592*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!batchnorm_5/AssignMovingAvg/sub/x?
batchnorm_5/AssignMovingAvg/subSub*batchnorm_5/AssignMovingAvg/sub/x:output:0batchnorm_5/Const_2:output:0*
T0*4
_class*
(&loc:@batchnorm_5/AssignMovingAvg/27592*
_output_shapes
: 2!
batchnorm_5/AssignMovingAvg/sub?
*batchnorm_5/AssignMovingAvg/ReadVariableOpReadVariableOp!batchnorm_5_assignmovingavg_27592*
_output_shapes	
:?*
dtype02,
*batchnorm_5/AssignMovingAvg/ReadVariableOp?
!batchnorm_5/AssignMovingAvg/sub_1Sub2batchnorm_5/AssignMovingAvg/ReadVariableOp:value:0)batchnorm_5/FusedBatchNormV3:batch_mean:0*
T0*4
_class*
(&loc:@batchnorm_5/AssignMovingAvg/27592*
_output_shapes	
:?2#
!batchnorm_5/AssignMovingAvg/sub_1?
batchnorm_5/AssignMovingAvg/mulMul%batchnorm_5/AssignMovingAvg/sub_1:z:0#batchnorm_5/AssignMovingAvg/sub:z:0*
T0*4
_class*
(&loc:@batchnorm_5/AssignMovingAvg/27592*
_output_shapes	
:?2!
batchnorm_5/AssignMovingAvg/mul?
/batchnorm_5/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batchnorm_5_assignmovingavg_27592#batchnorm_5/AssignMovingAvg/mul:z:0+^batchnorm_5/AssignMovingAvg/ReadVariableOp*4
_class*
(&loc:@batchnorm_5/AssignMovingAvg/27592*
_output_shapes
 *
dtype021
/batchnorm_5/AssignMovingAvg/AssignSubVariableOp?
#batchnorm_5/AssignMovingAvg_1/sub/xConst*6
_class,
*(loc:@batchnorm_5/AssignMovingAvg_1/27599*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#batchnorm_5/AssignMovingAvg_1/sub/x?
!batchnorm_5/AssignMovingAvg_1/subSub,batchnorm_5/AssignMovingAvg_1/sub/x:output:0batchnorm_5/Const_2:output:0*
T0*6
_class,
*(loc:@batchnorm_5/AssignMovingAvg_1/27599*
_output_shapes
: 2#
!batchnorm_5/AssignMovingAvg_1/sub?
,batchnorm_5/AssignMovingAvg_1/ReadVariableOpReadVariableOp#batchnorm_5_assignmovingavg_1_27599*
_output_shapes	
:?*
dtype02.
,batchnorm_5/AssignMovingAvg_1/ReadVariableOp?
#batchnorm_5/AssignMovingAvg_1/sub_1Sub4batchnorm_5/AssignMovingAvg_1/ReadVariableOp:value:0-batchnorm_5/FusedBatchNormV3:batch_variance:0*
T0*6
_class,
*(loc:@batchnorm_5/AssignMovingAvg_1/27599*
_output_shapes	
:?2%
#batchnorm_5/AssignMovingAvg_1/sub_1?
!batchnorm_5/AssignMovingAvg_1/mulMul'batchnorm_5/AssignMovingAvg_1/sub_1:z:0%batchnorm_5/AssignMovingAvg_1/sub:z:0*
T0*6
_class,
*(loc:@batchnorm_5/AssignMovingAvg_1/27599*
_output_shapes	
:?2#
!batchnorm_5/AssignMovingAvg_1/mul?
1batchnorm_5/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp#batchnorm_5_assignmovingavg_1_27599%batchnorm_5/AssignMovingAvg_1/mul:z:0-^batchnorm_5/AssignMovingAvg_1/ReadVariableOp*6
_class,
*(loc:@batchnorm_5/AssignMovingAvg_1/27599*
_output_shapes
 *
dtype023
1batchnorm_5/AssignMovingAvg_1/AssignSubVariableOp?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2D batchnorm_5/FusedBatchNormV3:y:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_6/BiasAddy
conv2d_6/EluEluconv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_6/Eluv
batchnorm_6/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
batchnorm_6/LogicalAnd/xv
batchnorm_6/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
batchnorm_6/LogicalAnd/y?
batchnorm_6/LogicalAnd
LogicalAnd!batchnorm_6/LogicalAnd/x:output:0!batchnorm_6/LogicalAnd/y:output:0*
_output_shapes
: 2
batchnorm_6/LogicalAnd?
batchnorm_6/ReadVariableOpReadVariableOp#batchnorm_6_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm_6/ReadVariableOp?
batchnorm_6/ReadVariableOp_1ReadVariableOp%batchnorm_6_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm_6/ReadVariableOp_1i
batchnorm_6/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
batchnorm_6/Constm
batchnorm_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
batchnorm_6/Const_1?
batchnorm_6/FusedBatchNormV3FusedBatchNormV3conv2d_6/Elu:activations:0"batchnorm_6/ReadVariableOp:value:0$batchnorm_6/ReadVariableOp_1:value:0batchnorm_6/Const:output:0batchnorm_6/Const_1:output:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:2
batchnorm_6/FusedBatchNormV3o
batchnorm_6/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
batchnorm_6/Const_2?
!batchnorm_6/AssignMovingAvg/sub/xConst*4
_class*
(&loc:@batchnorm_6/AssignMovingAvg/27629*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!batchnorm_6/AssignMovingAvg/sub/x?
batchnorm_6/AssignMovingAvg/subSub*batchnorm_6/AssignMovingAvg/sub/x:output:0batchnorm_6/Const_2:output:0*
T0*4
_class*
(&loc:@batchnorm_6/AssignMovingAvg/27629*
_output_shapes
: 2!
batchnorm_6/AssignMovingAvg/sub?
*batchnorm_6/AssignMovingAvg/ReadVariableOpReadVariableOp!batchnorm_6_assignmovingavg_27629*
_output_shapes	
:?*
dtype02,
*batchnorm_6/AssignMovingAvg/ReadVariableOp?
!batchnorm_6/AssignMovingAvg/sub_1Sub2batchnorm_6/AssignMovingAvg/ReadVariableOp:value:0)batchnorm_6/FusedBatchNormV3:batch_mean:0*
T0*4
_class*
(&loc:@batchnorm_6/AssignMovingAvg/27629*
_output_shapes	
:?2#
!batchnorm_6/AssignMovingAvg/sub_1?
batchnorm_6/AssignMovingAvg/mulMul%batchnorm_6/AssignMovingAvg/sub_1:z:0#batchnorm_6/AssignMovingAvg/sub:z:0*
T0*4
_class*
(&loc:@batchnorm_6/AssignMovingAvg/27629*
_output_shapes	
:?2!
batchnorm_6/AssignMovingAvg/mul?
/batchnorm_6/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batchnorm_6_assignmovingavg_27629#batchnorm_6/AssignMovingAvg/mul:z:0+^batchnorm_6/AssignMovingAvg/ReadVariableOp*4
_class*
(&loc:@batchnorm_6/AssignMovingAvg/27629*
_output_shapes
 *
dtype021
/batchnorm_6/AssignMovingAvg/AssignSubVariableOp?
#batchnorm_6/AssignMovingAvg_1/sub/xConst*6
_class,
*(loc:@batchnorm_6/AssignMovingAvg_1/27636*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#batchnorm_6/AssignMovingAvg_1/sub/x?
!batchnorm_6/AssignMovingAvg_1/subSub,batchnorm_6/AssignMovingAvg_1/sub/x:output:0batchnorm_6/Const_2:output:0*
T0*6
_class,
*(loc:@batchnorm_6/AssignMovingAvg_1/27636*
_output_shapes
: 2#
!batchnorm_6/AssignMovingAvg_1/sub?
,batchnorm_6/AssignMovingAvg_1/ReadVariableOpReadVariableOp#batchnorm_6_assignmovingavg_1_27636*
_output_shapes	
:?*
dtype02.
,batchnorm_6/AssignMovingAvg_1/ReadVariableOp?
#batchnorm_6/AssignMovingAvg_1/sub_1Sub4batchnorm_6/AssignMovingAvg_1/ReadVariableOp:value:0-batchnorm_6/FusedBatchNormV3:batch_variance:0*
T0*6
_class,
*(loc:@batchnorm_6/AssignMovingAvg_1/27636*
_output_shapes	
:?2%
#batchnorm_6/AssignMovingAvg_1/sub_1?
!batchnorm_6/AssignMovingAvg_1/mulMul'batchnorm_6/AssignMovingAvg_1/sub_1:z:0%batchnorm_6/AssignMovingAvg_1/sub:z:0*
T0*6
_class,
*(loc:@batchnorm_6/AssignMovingAvg_1/27636*
_output_shapes	
:?2#
!batchnorm_6/AssignMovingAvg_1/mul?
1batchnorm_6/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp#batchnorm_6_assignmovingavg_1_27636%batchnorm_6/AssignMovingAvg_1/mul:z:0-^batchnorm_6/AssignMovingAvg_1/ReadVariableOp*6
_class,
*(loc:@batchnorm_6/AssignMovingAvg_1/27636*
_output_shapes
 *
dtype023
1batchnorm_6/AssignMovingAvg_1/AssignSubVariableOp?
maxpool2d_3/MaxPoolMaxPool batchnorm_6/FusedBatchNormV3:y:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
maxpool2d_3/MaxPoolu
dropout_3/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_3/dropout/rate~
dropout_3/dropout/ShapeShapemaxpool2d_3/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_3/dropout/Shape?
$dropout_3/dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$dropout_3/dropout/random_uniform/min?
$dropout_3/dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$dropout_3/dropout/random_uniform/max?
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype020
.dropout_3/dropout/random_uniform/RandomUniform?
$dropout_3/dropout/random_uniform/subSub-dropout_3/dropout/random_uniform/max:output:0-dropout_3/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2&
$dropout_3/dropout/random_uniform/sub?
$dropout_3/dropout/random_uniform/mulMul7dropout_3/dropout/random_uniform/RandomUniform:output:0(dropout_3/dropout/random_uniform/sub:z:0*
T0*0
_output_shapes
:??????????2&
$dropout_3/dropout/random_uniform/mul?
 dropout_3/dropout/random_uniformAdd(dropout_3/dropout/random_uniform/mul:z:0-dropout_3/dropout/random_uniform/min:output:0*
T0*0
_output_shapes
:??????????2"
 dropout_3/dropout/random_uniformw
dropout_3/dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_3/dropout/sub/x?
dropout_3/dropout/subSub dropout_3/dropout/sub/x:output:0dropout_3/dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout_3/dropout/sub
dropout_3/dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_3/dropout/truediv/x?
dropout_3/dropout/truedivRealDiv$dropout_3/dropout/truediv/x:output:0dropout_3/dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout_3/dropout/truediv?
dropout_3/dropout/GreaterEqualGreaterEqual$dropout_3/dropout/random_uniform:z:0dropout_3/dropout/rate:output:0*
T0*0
_output_shapes
:??????????2 
dropout_3/dropout/GreaterEqual?
dropout_3/dropout/mulMulmaxpool2d_3/MaxPool:output:0dropout_3/dropout/truediv:z:0*
T0*0
_output_shapes
:??????????2
dropout_3/dropout/mul?
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
dropout_3/dropout/Cast?
dropout_3/dropout/mul_1Muldropout_3/dropout/mul:z:0dropout_3/dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
dropout_3/dropout/mul_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? $  2
flatten/Const?
flatten/ReshapeReshapedropout_3/dropout/mul_1:z:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????H2
flatten/Reshape?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
?H?*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddn
dense_1/EluEludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_1/Eluv
batchnorm_7/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
batchnorm_7/LogicalAnd/xv
batchnorm_7/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
batchnorm_7/LogicalAnd/y?
batchnorm_7/LogicalAnd
LogicalAnd!batchnorm_7/LogicalAnd/x:output:0!batchnorm_7/LogicalAnd/y:output:0*
_output_shapes
: 2
batchnorm_7/LogicalAnd?
*batchnorm_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2,
*batchnorm_7/moments/mean/reduction_indices?
batchnorm_7/moments/meanMeandense_1/Elu:activations:03batchnorm_7/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
batchnorm_7/moments/mean?
 batchnorm_7/moments/StopGradientStopGradient!batchnorm_7/moments/mean:output:0*
T0*
_output_shapes
:	?2"
 batchnorm_7/moments/StopGradient?
%batchnorm_7/moments/SquaredDifferenceSquaredDifferencedense_1/Elu:activations:0)batchnorm_7/moments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2'
%batchnorm_7/moments/SquaredDifference?
.batchnorm_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 20
.batchnorm_7/moments/variance/reduction_indices?
batchnorm_7/moments/varianceMean)batchnorm_7/moments/SquaredDifference:z:07batchnorm_7/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
batchnorm_7/moments/variance?
batchnorm_7/moments/SqueezeSqueeze!batchnorm_7/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
batchnorm_7/moments/Squeeze?
batchnorm_7/moments/Squeeze_1Squeeze%batchnorm_7/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
batchnorm_7/moments/Squeeze_1?
!batchnorm_7/AssignMovingAvg/decayConst*4
_class*
(&loc:@batchnorm_7/AssignMovingAvg/27680*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!batchnorm_7/AssignMovingAvg/decay?
*batchnorm_7/AssignMovingAvg/ReadVariableOpReadVariableOp!batchnorm_7_assignmovingavg_27680*
_output_shapes	
:?*
dtype02,
*batchnorm_7/AssignMovingAvg/ReadVariableOp?
batchnorm_7/AssignMovingAvg/subSub2batchnorm_7/AssignMovingAvg/ReadVariableOp:value:0$batchnorm_7/moments/Squeeze:output:0*
T0*4
_class*
(&loc:@batchnorm_7/AssignMovingAvg/27680*
_output_shapes	
:?2!
batchnorm_7/AssignMovingAvg/sub?
batchnorm_7/AssignMovingAvg/mulMul#batchnorm_7/AssignMovingAvg/sub:z:0*batchnorm_7/AssignMovingAvg/decay:output:0*
T0*4
_class*
(&loc:@batchnorm_7/AssignMovingAvg/27680*
_output_shapes	
:?2!
batchnorm_7/AssignMovingAvg/mul?
/batchnorm_7/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batchnorm_7_assignmovingavg_27680#batchnorm_7/AssignMovingAvg/mul:z:0+^batchnorm_7/AssignMovingAvg/ReadVariableOp*4
_class*
(&loc:@batchnorm_7/AssignMovingAvg/27680*
_output_shapes
 *
dtype021
/batchnorm_7/AssignMovingAvg/AssignSubVariableOp?
#batchnorm_7/AssignMovingAvg_1/decayConst*6
_class,
*(loc:@batchnorm_7/AssignMovingAvg_1/27686*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#batchnorm_7/AssignMovingAvg_1/decay?
,batchnorm_7/AssignMovingAvg_1/ReadVariableOpReadVariableOp#batchnorm_7_assignmovingavg_1_27686*
_output_shapes	
:?*
dtype02.
,batchnorm_7/AssignMovingAvg_1/ReadVariableOp?
!batchnorm_7/AssignMovingAvg_1/subSub4batchnorm_7/AssignMovingAvg_1/ReadVariableOp:value:0&batchnorm_7/moments/Squeeze_1:output:0*
T0*6
_class,
*(loc:@batchnorm_7/AssignMovingAvg_1/27686*
_output_shapes	
:?2#
!batchnorm_7/AssignMovingAvg_1/sub?
!batchnorm_7/AssignMovingAvg_1/mulMul%batchnorm_7/AssignMovingAvg_1/sub:z:0,batchnorm_7/AssignMovingAvg_1/decay:output:0*
T0*6
_class,
*(loc:@batchnorm_7/AssignMovingAvg_1/27686*
_output_shapes	
:?2#
!batchnorm_7/AssignMovingAvg_1/mul?
1batchnorm_7/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp#batchnorm_7_assignmovingavg_1_27686%batchnorm_7/AssignMovingAvg_1/mul:z:0-^batchnorm_7/AssignMovingAvg_1/ReadVariableOp*6
_class,
*(loc:@batchnorm_7/AssignMovingAvg_1/27686*
_output_shapes
 *
dtype023
1batchnorm_7/AssignMovingAvg_1/AssignSubVariableOp
batchnorm_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm_7/batchnorm/add/y?
batchnorm_7/batchnorm/addAddV2&batchnorm_7/moments/Squeeze_1:output:0$batchnorm_7/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm_7/batchnorm/add?
batchnorm_7/batchnorm/RsqrtRsqrtbatchnorm_7/batchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm_7/batchnorm/Rsqrt?
(batchnorm_7/batchnorm/mul/ReadVariableOpReadVariableOp1batchnorm_7_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(batchnorm_7/batchnorm/mul/ReadVariableOp?
batchnorm_7/batchnorm/mulMulbatchnorm_7/batchnorm/Rsqrt:y:00batchnorm_7/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm_7/batchnorm/mul?
batchnorm_7/batchnorm/mul_1Muldense_1/Elu:activations:0batchnorm_7/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm_7/batchnorm/mul_1?
batchnorm_7/batchnorm/mul_2Mul$batchnorm_7/moments/Squeeze:output:0batchnorm_7/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm_7/batchnorm/mul_2?
$batchnorm_7/batchnorm/ReadVariableOpReadVariableOp-batchnorm_7_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batchnorm_7/batchnorm/ReadVariableOp?
batchnorm_7/batchnorm/subSub,batchnorm_7/batchnorm/ReadVariableOp:value:0batchnorm_7/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm_7/batchnorm/sub?
batchnorm_7/batchnorm/add_1AddV2batchnorm_7/batchnorm/mul_1:z:0batchnorm_7/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm_7/batchnorm/add_1u
dropout_4/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *???2
dropout_4/dropout/rate?
dropout_4/dropout/ShapeShapebatchnorm_7/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
dropout_4/dropout/Shape?
$dropout_4/dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$dropout_4/dropout/random_uniform/min?
$dropout_4/dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$dropout_4/dropout/random_uniform/max?
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype020
.dropout_4/dropout/random_uniform/RandomUniform?
$dropout_4/dropout/random_uniform/subSub-dropout_4/dropout/random_uniform/max:output:0-dropout_4/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2&
$dropout_4/dropout/random_uniform/sub?
$dropout_4/dropout/random_uniform/mulMul7dropout_4/dropout/random_uniform/RandomUniform:output:0(dropout_4/dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:??????????2&
$dropout_4/dropout/random_uniform/mul?
 dropout_4/dropout/random_uniformAdd(dropout_4/dropout/random_uniform/mul:z:0-dropout_4/dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:??????????2"
 dropout_4/dropout/random_uniformw
dropout_4/dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_4/dropout/sub/x?
dropout_4/dropout/subSub dropout_4/dropout/sub/x:output:0dropout_4/dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout_4/dropout/sub
dropout_4/dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_4/dropout/truediv/x?
dropout_4/dropout/truedivRealDiv$dropout_4/dropout/truediv/x:output:0dropout_4/dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout_4/dropout/truediv?
dropout_4/dropout/GreaterEqualGreaterEqual$dropout_4/dropout/random_uniform:z:0dropout_4/dropout/rate:output:0*
T0*(
_output_shapes
:??????????2 
dropout_4/dropout/GreaterEqual?
dropout_4/dropout/mulMulbatchnorm_7/batchnorm/add_1:z:0dropout_4/dropout/truediv:z:0*
T0*(
_output_shapes
:??????????2
dropout_4/dropout/mul?
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_4/dropout/Cast?
dropout_4/dropout/mul_1Muldropout_4/dropout/mul:z:0dropout_4/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_4/dropout/mul_1?
out_layer/MatMul/ReadVariableOpReadVariableOp(out_layer_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02!
out_layer/MatMul/ReadVariableOp?
out_layer/MatMulMatMuldropout_4/dropout/mul_1:z:0'out_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
out_layer/MatMul?
 out_layer/BiasAdd/ReadVariableOpReadVariableOp)out_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 out_layer/BiasAdd/ReadVariableOp?
out_layer/BiasAddBiasAddout_layer/MatMul:product:0(out_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
out_layer/BiasAdd
out_layer/SoftmaxSoftmaxout_layer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
out_layer/Softmax?
IdentityIdentityout_layer/Softmax:softmax:00^batchnorm_1/AssignMovingAvg/AssignSubVariableOp+^batchnorm_1/AssignMovingAvg/ReadVariableOp2^batchnorm_1/AssignMovingAvg_1/AssignSubVariableOp-^batchnorm_1/AssignMovingAvg_1/ReadVariableOp^batchnorm_1/ReadVariableOp^batchnorm_1/ReadVariableOp_10^batchnorm_2/AssignMovingAvg/AssignSubVariableOp+^batchnorm_2/AssignMovingAvg/ReadVariableOp2^batchnorm_2/AssignMovingAvg_1/AssignSubVariableOp-^batchnorm_2/AssignMovingAvg_1/ReadVariableOp^batchnorm_2/ReadVariableOp^batchnorm_2/ReadVariableOp_10^batchnorm_3/AssignMovingAvg/AssignSubVariableOp+^batchnorm_3/AssignMovingAvg/ReadVariableOp2^batchnorm_3/AssignMovingAvg_1/AssignSubVariableOp-^batchnorm_3/AssignMovingAvg_1/ReadVariableOp^batchnorm_3/ReadVariableOp^batchnorm_3/ReadVariableOp_10^batchnorm_4/AssignMovingAvg/AssignSubVariableOp+^batchnorm_4/AssignMovingAvg/ReadVariableOp2^batchnorm_4/AssignMovingAvg_1/AssignSubVariableOp-^batchnorm_4/AssignMovingAvg_1/ReadVariableOp^batchnorm_4/ReadVariableOp^batchnorm_4/ReadVariableOp_10^batchnorm_5/AssignMovingAvg/AssignSubVariableOp+^batchnorm_5/AssignMovingAvg/ReadVariableOp2^batchnorm_5/AssignMovingAvg_1/AssignSubVariableOp-^batchnorm_5/AssignMovingAvg_1/ReadVariableOp^batchnorm_5/ReadVariableOp^batchnorm_5/ReadVariableOp_10^batchnorm_6/AssignMovingAvg/AssignSubVariableOp+^batchnorm_6/AssignMovingAvg/ReadVariableOp2^batchnorm_6/AssignMovingAvg_1/AssignSubVariableOp-^batchnorm_6/AssignMovingAvg_1/ReadVariableOp^batchnorm_6/ReadVariableOp^batchnorm_6/ReadVariableOp_10^batchnorm_7/AssignMovingAvg/AssignSubVariableOp+^batchnorm_7/AssignMovingAvg/ReadVariableOp2^batchnorm_7/AssignMovingAvg_1/AssignSubVariableOp-^batchnorm_7/AssignMovingAvg_1/ReadVariableOp%^batchnorm_7/batchnorm/ReadVariableOp)^batchnorm_7/batchnorm/mul/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp!^out_layer/BiasAdd/ReadVariableOp ^out_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????00::::::::::::::::::::::::::::::::::::::::::::2b
/batchnorm_1/AssignMovingAvg/AssignSubVariableOp/batchnorm_1/AssignMovingAvg/AssignSubVariableOp2X
*batchnorm_1/AssignMovingAvg/ReadVariableOp*batchnorm_1/AssignMovingAvg/ReadVariableOp2f
1batchnorm_1/AssignMovingAvg_1/AssignSubVariableOp1batchnorm_1/AssignMovingAvg_1/AssignSubVariableOp2\
,batchnorm_1/AssignMovingAvg_1/ReadVariableOp,batchnorm_1/AssignMovingAvg_1/ReadVariableOp28
batchnorm_1/ReadVariableOpbatchnorm_1/ReadVariableOp2<
batchnorm_1/ReadVariableOp_1batchnorm_1/ReadVariableOp_12b
/batchnorm_2/AssignMovingAvg/AssignSubVariableOp/batchnorm_2/AssignMovingAvg/AssignSubVariableOp2X
*batchnorm_2/AssignMovingAvg/ReadVariableOp*batchnorm_2/AssignMovingAvg/ReadVariableOp2f
1batchnorm_2/AssignMovingAvg_1/AssignSubVariableOp1batchnorm_2/AssignMovingAvg_1/AssignSubVariableOp2\
,batchnorm_2/AssignMovingAvg_1/ReadVariableOp,batchnorm_2/AssignMovingAvg_1/ReadVariableOp28
batchnorm_2/ReadVariableOpbatchnorm_2/ReadVariableOp2<
batchnorm_2/ReadVariableOp_1batchnorm_2/ReadVariableOp_12b
/batchnorm_3/AssignMovingAvg/AssignSubVariableOp/batchnorm_3/AssignMovingAvg/AssignSubVariableOp2X
*batchnorm_3/AssignMovingAvg/ReadVariableOp*batchnorm_3/AssignMovingAvg/ReadVariableOp2f
1batchnorm_3/AssignMovingAvg_1/AssignSubVariableOp1batchnorm_3/AssignMovingAvg_1/AssignSubVariableOp2\
,batchnorm_3/AssignMovingAvg_1/ReadVariableOp,batchnorm_3/AssignMovingAvg_1/ReadVariableOp28
batchnorm_3/ReadVariableOpbatchnorm_3/ReadVariableOp2<
batchnorm_3/ReadVariableOp_1batchnorm_3/ReadVariableOp_12b
/batchnorm_4/AssignMovingAvg/AssignSubVariableOp/batchnorm_4/AssignMovingAvg/AssignSubVariableOp2X
*batchnorm_4/AssignMovingAvg/ReadVariableOp*batchnorm_4/AssignMovingAvg/ReadVariableOp2f
1batchnorm_4/AssignMovingAvg_1/AssignSubVariableOp1batchnorm_4/AssignMovingAvg_1/AssignSubVariableOp2\
,batchnorm_4/AssignMovingAvg_1/ReadVariableOp,batchnorm_4/AssignMovingAvg_1/ReadVariableOp28
batchnorm_4/ReadVariableOpbatchnorm_4/ReadVariableOp2<
batchnorm_4/ReadVariableOp_1batchnorm_4/ReadVariableOp_12b
/batchnorm_5/AssignMovingAvg/AssignSubVariableOp/batchnorm_5/AssignMovingAvg/AssignSubVariableOp2X
*batchnorm_5/AssignMovingAvg/ReadVariableOp*batchnorm_5/AssignMovingAvg/ReadVariableOp2f
1batchnorm_5/AssignMovingAvg_1/AssignSubVariableOp1batchnorm_5/AssignMovingAvg_1/AssignSubVariableOp2\
,batchnorm_5/AssignMovingAvg_1/ReadVariableOp,batchnorm_5/AssignMovingAvg_1/ReadVariableOp28
batchnorm_5/ReadVariableOpbatchnorm_5/ReadVariableOp2<
batchnorm_5/ReadVariableOp_1batchnorm_5/ReadVariableOp_12b
/batchnorm_6/AssignMovingAvg/AssignSubVariableOp/batchnorm_6/AssignMovingAvg/AssignSubVariableOp2X
*batchnorm_6/AssignMovingAvg/ReadVariableOp*batchnorm_6/AssignMovingAvg/ReadVariableOp2f
1batchnorm_6/AssignMovingAvg_1/AssignSubVariableOp1batchnorm_6/AssignMovingAvg_1/AssignSubVariableOp2\
,batchnorm_6/AssignMovingAvg_1/ReadVariableOp,batchnorm_6/AssignMovingAvg_1/ReadVariableOp28
batchnorm_6/ReadVariableOpbatchnorm_6/ReadVariableOp2<
batchnorm_6/ReadVariableOp_1batchnorm_6/ReadVariableOp_12b
/batchnorm_7/AssignMovingAvg/AssignSubVariableOp/batchnorm_7/AssignMovingAvg/AssignSubVariableOp2X
*batchnorm_7/AssignMovingAvg/ReadVariableOp*batchnorm_7/AssignMovingAvg/ReadVariableOp2f
1batchnorm_7/AssignMovingAvg_1/AssignSubVariableOp1batchnorm_7/AssignMovingAvg_1/AssignSubVariableOp2\
,batchnorm_7/AssignMovingAvg_1/ReadVariableOp,batchnorm_7/AssignMovingAvg_1/ReadVariableOp2L
$batchnorm_7/batchnorm/ReadVariableOp$batchnorm_7/batchnorm/ReadVariableOp2T
(batchnorm_7/batchnorm/mul/ReadVariableOp(batchnorm_7/batchnorm/mul/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2D
 out_layer/BiasAdd/ReadVariableOp out_layer/BiasAdd/ReadVariableOp2B
out_layer/MatMul/ReadVariableOpout_layer/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
??
?0
__inference__traced_save_29642
file_prefix0
,savev2_conv2d_1_1_kernel_read_readvariableop.
*savev2_conv2d_1_1_bias_read_readvariableop2
.savev2_batchnorm_1_1_gamma_read_readvariableop1
-savev2_batchnorm_1_1_beta_read_readvariableop8
4savev2_batchnorm_1_1_moving_mean_read_readvariableop<
8savev2_batchnorm_1_1_moving_variance_read_readvariableop0
,savev2_conv2d_2_1_kernel_read_readvariableop.
*savev2_conv2d_2_1_bias_read_readvariableop2
.savev2_batchnorm_2_1_gamma_read_readvariableop1
-savev2_batchnorm_2_1_beta_read_readvariableop8
4savev2_batchnorm_2_1_moving_mean_read_readvariableop<
8savev2_batchnorm_2_1_moving_variance_read_readvariableop0
,savev2_conv2d_3_1_kernel_read_readvariableop.
*savev2_conv2d_3_1_bias_read_readvariableop2
.savev2_batchnorm_3_1_gamma_read_readvariableop1
-savev2_batchnorm_3_1_beta_read_readvariableop8
4savev2_batchnorm_3_1_moving_mean_read_readvariableop<
8savev2_batchnorm_3_1_moving_variance_read_readvariableop0
,savev2_conv2d_4_1_kernel_read_readvariableop.
*savev2_conv2d_4_1_bias_read_readvariableop2
.savev2_batchnorm_4_1_gamma_read_readvariableop1
-savev2_batchnorm_4_1_beta_read_readvariableop8
4savev2_batchnorm_4_1_moving_mean_read_readvariableop<
8savev2_batchnorm_4_1_moving_variance_read_readvariableop0
,savev2_conv2d_5_1_kernel_read_readvariableop.
*savev2_conv2d_5_1_bias_read_readvariableop2
.savev2_batchnorm_5_1_gamma_read_readvariableop1
-savev2_batchnorm_5_1_beta_read_readvariableop8
4savev2_batchnorm_5_1_moving_mean_read_readvariableop<
8savev2_batchnorm_5_1_moving_variance_read_readvariableop0
,savev2_conv2d_6_1_kernel_read_readvariableop.
*savev2_conv2d_6_1_bias_read_readvariableop2
.savev2_batchnorm_6_1_gamma_read_readvariableop1
-savev2_batchnorm_6_1_beta_read_readvariableop8
4savev2_batchnorm_6_1_moving_mean_read_readvariableop<
8savev2_batchnorm_6_1_moving_variance_read_readvariableop/
+savev2_dense_1_1_kernel_read_readvariableop-
)savev2_dense_1_1_bias_read_readvariableop2
.savev2_batchnorm_7_1_gamma_read_readvariableop1
-savev2_batchnorm_7_1_beta_read_readvariableop8
4savev2_batchnorm_7_1_moving_mean_read_readvariableop<
8savev2_batchnorm_7_1_moving_variance_read_readvariableop1
-savev2_out_layer_1_kernel_read_readvariableop/
+savev2_out_layer_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop7
3savev2_adam_conv2d_1_1_kernel_m_read_readvariableop5
1savev2_adam_conv2d_1_1_bias_m_read_readvariableop9
5savev2_adam_batchnorm_1_1_gamma_m_read_readvariableop8
4savev2_adam_batchnorm_1_1_beta_m_read_readvariableop7
3savev2_adam_conv2d_2_1_kernel_m_read_readvariableop5
1savev2_adam_conv2d_2_1_bias_m_read_readvariableop9
5savev2_adam_batchnorm_2_1_gamma_m_read_readvariableop8
4savev2_adam_batchnorm_2_1_beta_m_read_readvariableop7
3savev2_adam_conv2d_3_1_kernel_m_read_readvariableop5
1savev2_adam_conv2d_3_1_bias_m_read_readvariableop9
5savev2_adam_batchnorm_3_1_gamma_m_read_readvariableop8
4savev2_adam_batchnorm_3_1_beta_m_read_readvariableop7
3savev2_adam_conv2d_4_1_kernel_m_read_readvariableop5
1savev2_adam_conv2d_4_1_bias_m_read_readvariableop9
5savev2_adam_batchnorm_4_1_gamma_m_read_readvariableop8
4savev2_adam_batchnorm_4_1_beta_m_read_readvariableop7
3savev2_adam_conv2d_5_1_kernel_m_read_readvariableop5
1savev2_adam_conv2d_5_1_bias_m_read_readvariableop9
5savev2_adam_batchnorm_5_1_gamma_m_read_readvariableop8
4savev2_adam_batchnorm_5_1_beta_m_read_readvariableop7
3savev2_adam_conv2d_6_1_kernel_m_read_readvariableop5
1savev2_adam_conv2d_6_1_bias_m_read_readvariableop9
5savev2_adam_batchnorm_6_1_gamma_m_read_readvariableop8
4savev2_adam_batchnorm_6_1_beta_m_read_readvariableop6
2savev2_adam_dense_1_1_kernel_m_read_readvariableop4
0savev2_adam_dense_1_1_bias_m_read_readvariableop9
5savev2_adam_batchnorm_7_1_gamma_m_read_readvariableop8
4savev2_adam_batchnorm_7_1_beta_m_read_readvariableop8
4savev2_adam_out_layer_1_kernel_m_read_readvariableop6
2savev2_adam_out_layer_1_bias_m_read_readvariableop7
3savev2_adam_conv2d_1_1_kernel_v_read_readvariableop5
1savev2_adam_conv2d_1_1_bias_v_read_readvariableop9
5savev2_adam_batchnorm_1_1_gamma_v_read_readvariableop8
4savev2_adam_batchnorm_1_1_beta_v_read_readvariableop7
3savev2_adam_conv2d_2_1_kernel_v_read_readvariableop5
1savev2_adam_conv2d_2_1_bias_v_read_readvariableop9
5savev2_adam_batchnorm_2_1_gamma_v_read_readvariableop8
4savev2_adam_batchnorm_2_1_beta_v_read_readvariableop7
3savev2_adam_conv2d_3_1_kernel_v_read_readvariableop5
1savev2_adam_conv2d_3_1_bias_v_read_readvariableop9
5savev2_adam_batchnorm_3_1_gamma_v_read_readvariableop8
4savev2_adam_batchnorm_3_1_beta_v_read_readvariableop7
3savev2_adam_conv2d_4_1_kernel_v_read_readvariableop5
1savev2_adam_conv2d_4_1_bias_v_read_readvariableop9
5savev2_adam_batchnorm_4_1_gamma_v_read_readvariableop8
4savev2_adam_batchnorm_4_1_beta_v_read_readvariableop7
3savev2_adam_conv2d_5_1_kernel_v_read_readvariableop5
1savev2_adam_conv2d_5_1_bias_v_read_readvariableop9
5savev2_adam_batchnorm_5_1_gamma_v_read_readvariableop8
4savev2_adam_batchnorm_5_1_beta_v_read_readvariableop7
3savev2_adam_conv2d_6_1_kernel_v_read_readvariableop5
1savev2_adam_conv2d_6_1_bias_v_read_readvariableop9
5savev2_adam_batchnorm_6_1_gamma_v_read_readvariableop8
4savev2_adam_batchnorm_6_1_beta_v_read_readvariableop6
2savev2_adam_dense_1_1_kernel_v_read_readvariableop4
0savev2_adam_dense_1_1_bias_v_read_readvariableop9
5savev2_adam_batchnorm_7_1_gamma_v_read_readvariableop8
4savev2_adam_batchnorm_7_1_beta_v_read_readvariableop8
4savev2_adam_out_layer_1_kernel_v_read_readvariableop6
2savev2_adam_out_layer_1_bias_v_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_679b57378e5548c4a29eff4c17641bcb/part2
StringJoin/inputs_1?

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?>
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:o*
dtype0*?=
value?=B?=oB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:o*
dtype0*?
value?B?oB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?.
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_conv2d_1_1_kernel_read_readvariableop*savev2_conv2d_1_1_bias_read_readvariableop.savev2_batchnorm_1_1_gamma_read_readvariableop-savev2_batchnorm_1_1_beta_read_readvariableop4savev2_batchnorm_1_1_moving_mean_read_readvariableop8savev2_batchnorm_1_1_moving_variance_read_readvariableop,savev2_conv2d_2_1_kernel_read_readvariableop*savev2_conv2d_2_1_bias_read_readvariableop.savev2_batchnorm_2_1_gamma_read_readvariableop-savev2_batchnorm_2_1_beta_read_readvariableop4savev2_batchnorm_2_1_moving_mean_read_readvariableop8savev2_batchnorm_2_1_moving_variance_read_readvariableop,savev2_conv2d_3_1_kernel_read_readvariableop*savev2_conv2d_3_1_bias_read_readvariableop.savev2_batchnorm_3_1_gamma_read_readvariableop-savev2_batchnorm_3_1_beta_read_readvariableop4savev2_batchnorm_3_1_moving_mean_read_readvariableop8savev2_batchnorm_3_1_moving_variance_read_readvariableop,savev2_conv2d_4_1_kernel_read_readvariableop*savev2_conv2d_4_1_bias_read_readvariableop.savev2_batchnorm_4_1_gamma_read_readvariableop-savev2_batchnorm_4_1_beta_read_readvariableop4savev2_batchnorm_4_1_moving_mean_read_readvariableop8savev2_batchnorm_4_1_moving_variance_read_readvariableop,savev2_conv2d_5_1_kernel_read_readvariableop*savev2_conv2d_5_1_bias_read_readvariableop.savev2_batchnorm_5_1_gamma_read_readvariableop-savev2_batchnorm_5_1_beta_read_readvariableop4savev2_batchnorm_5_1_moving_mean_read_readvariableop8savev2_batchnorm_5_1_moving_variance_read_readvariableop,savev2_conv2d_6_1_kernel_read_readvariableop*savev2_conv2d_6_1_bias_read_readvariableop.savev2_batchnorm_6_1_gamma_read_readvariableop-savev2_batchnorm_6_1_beta_read_readvariableop4savev2_batchnorm_6_1_moving_mean_read_readvariableop8savev2_batchnorm_6_1_moving_variance_read_readvariableop+savev2_dense_1_1_kernel_read_readvariableop)savev2_dense_1_1_bias_read_readvariableop.savev2_batchnorm_7_1_gamma_read_readvariableop-savev2_batchnorm_7_1_beta_read_readvariableop4savev2_batchnorm_7_1_moving_mean_read_readvariableop8savev2_batchnorm_7_1_moving_variance_read_readvariableop-savev2_out_layer_1_kernel_read_readvariableop+savev2_out_layer_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop3savev2_adam_conv2d_1_1_kernel_m_read_readvariableop1savev2_adam_conv2d_1_1_bias_m_read_readvariableop5savev2_adam_batchnorm_1_1_gamma_m_read_readvariableop4savev2_adam_batchnorm_1_1_beta_m_read_readvariableop3savev2_adam_conv2d_2_1_kernel_m_read_readvariableop1savev2_adam_conv2d_2_1_bias_m_read_readvariableop5savev2_adam_batchnorm_2_1_gamma_m_read_readvariableop4savev2_adam_batchnorm_2_1_beta_m_read_readvariableop3savev2_adam_conv2d_3_1_kernel_m_read_readvariableop1savev2_adam_conv2d_3_1_bias_m_read_readvariableop5savev2_adam_batchnorm_3_1_gamma_m_read_readvariableop4savev2_adam_batchnorm_3_1_beta_m_read_readvariableop3savev2_adam_conv2d_4_1_kernel_m_read_readvariableop1savev2_adam_conv2d_4_1_bias_m_read_readvariableop5savev2_adam_batchnorm_4_1_gamma_m_read_readvariableop4savev2_adam_batchnorm_4_1_beta_m_read_readvariableop3savev2_adam_conv2d_5_1_kernel_m_read_readvariableop1savev2_adam_conv2d_5_1_bias_m_read_readvariableop5savev2_adam_batchnorm_5_1_gamma_m_read_readvariableop4savev2_adam_batchnorm_5_1_beta_m_read_readvariableop3savev2_adam_conv2d_6_1_kernel_m_read_readvariableop1savev2_adam_conv2d_6_1_bias_m_read_readvariableop5savev2_adam_batchnorm_6_1_gamma_m_read_readvariableop4savev2_adam_batchnorm_6_1_beta_m_read_readvariableop2savev2_adam_dense_1_1_kernel_m_read_readvariableop0savev2_adam_dense_1_1_bias_m_read_readvariableop5savev2_adam_batchnorm_7_1_gamma_m_read_readvariableop4savev2_adam_batchnorm_7_1_beta_m_read_readvariableop4savev2_adam_out_layer_1_kernel_m_read_readvariableop2savev2_adam_out_layer_1_bias_m_read_readvariableop3savev2_adam_conv2d_1_1_kernel_v_read_readvariableop1savev2_adam_conv2d_1_1_bias_v_read_readvariableop5savev2_adam_batchnorm_1_1_gamma_v_read_readvariableop4savev2_adam_batchnorm_1_1_beta_v_read_readvariableop3savev2_adam_conv2d_2_1_kernel_v_read_readvariableop1savev2_adam_conv2d_2_1_bias_v_read_readvariableop5savev2_adam_batchnorm_2_1_gamma_v_read_readvariableop4savev2_adam_batchnorm_2_1_beta_v_read_readvariableop3savev2_adam_conv2d_3_1_kernel_v_read_readvariableop1savev2_adam_conv2d_3_1_bias_v_read_readvariableop5savev2_adam_batchnorm_3_1_gamma_v_read_readvariableop4savev2_adam_batchnorm_3_1_beta_v_read_readvariableop3savev2_adam_conv2d_4_1_kernel_v_read_readvariableop1savev2_adam_conv2d_4_1_bias_v_read_readvariableop5savev2_adam_batchnorm_4_1_gamma_v_read_readvariableop4savev2_adam_batchnorm_4_1_beta_v_read_readvariableop3savev2_adam_conv2d_5_1_kernel_v_read_readvariableop1savev2_adam_conv2d_5_1_bias_v_read_readvariableop5savev2_adam_batchnorm_5_1_gamma_v_read_readvariableop4savev2_adam_batchnorm_5_1_beta_v_read_readvariableop3savev2_adam_conv2d_6_1_kernel_v_read_readvariableop1savev2_adam_conv2d_6_1_bias_v_read_readvariableop5savev2_adam_batchnorm_6_1_gamma_v_read_readvariableop4savev2_adam_batchnorm_6_1_beta_v_read_readvariableop2savev2_adam_dense_1_1_kernel_v_read_readvariableop0savev2_adam_dense_1_1_bias_v_read_readvariableop5savev2_adam_batchnorm_7_1_gamma_v_read_readvariableop4savev2_adam_batchnorm_7_1_beta_v_read_readvariableop4savev2_adam_out_layer_1_kernel_v_read_readvariableop2savev2_adam_out_layer_1_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *}
dtypess
q2o	2
SaveV2?
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names?
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity?

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :@:@:@:@:@:@:@@:@:@:@:@:@:@?:?:?:?:?:?:??:?:?:?:?:?:??:?:?:?:?:?:??:?:?:?:?:?:
?H?:?:?:?:?:?:	?:: : : : : : : :@:@:@:@:@@:@:@:@:@?:?:?:?:??:?:?:?:??:?:?:?:??:?:?:?:
?H?:?:?:?:	?::@:@:@:@:@@:@:@:@:@?:?:?:?:??:?:?:?:??:?:?:?:??:?:?:?:
?H?:?:?:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
?$
?
F__inference_batchnorm_4_layer_call_and_return_conditional_losses_25659

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_25644
assignmovingavg_1_25651
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *?p}?2	
Const_2?
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/25644*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/25644*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_25644*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/25644*
_output_shapes	
:?2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/25644*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_25644AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/25644*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/25651*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/25651*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_25651*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/25651*
_output_shapes	
:?2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/25651*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_25651AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/25651*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?
?
+__inference_batchnorm_1_layer_call_fn_28099

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_1_layer_call_and_return_conditional_losses_251882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?$
?
F__inference_batchnorm_3_layer_call_and_return_conditional_losses_28423

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_28408
assignmovingavg_1_28415
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *?p}?2	
Const_2?
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/28408*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/28408*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_28408*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/28408*
_output_shapes	
:?2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/28408*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_28408AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/28408*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/28415*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/28415*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_28415*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/28415*
_output_shapes	
:?2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/28415*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_28415AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/28415*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?~
?
?__inference_DCNN_layer_call_and_return_conditional_losses_27051

inputs+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2.
*batchnorm_1_statefulpartitionedcall_args_1.
*batchnorm_1_statefulpartitionedcall_args_2.
*batchnorm_1_statefulpartitionedcall_args_3.
*batchnorm_1_statefulpartitionedcall_args_4+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2.
*batchnorm_2_statefulpartitionedcall_args_1.
*batchnorm_2_statefulpartitionedcall_args_2.
*batchnorm_2_statefulpartitionedcall_args_3.
*batchnorm_2_statefulpartitionedcall_args_4+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_2.
*batchnorm_3_statefulpartitionedcall_args_1.
*batchnorm_3_statefulpartitionedcall_args_2.
*batchnorm_3_statefulpartitionedcall_args_3.
*batchnorm_3_statefulpartitionedcall_args_4+
'conv2d_4_statefulpartitionedcall_args_1+
'conv2d_4_statefulpartitionedcall_args_2.
*batchnorm_4_statefulpartitionedcall_args_1.
*batchnorm_4_statefulpartitionedcall_args_2.
*batchnorm_4_statefulpartitionedcall_args_3.
*batchnorm_4_statefulpartitionedcall_args_4+
'conv2d_5_statefulpartitionedcall_args_1+
'conv2d_5_statefulpartitionedcall_args_2.
*batchnorm_5_statefulpartitionedcall_args_1.
*batchnorm_5_statefulpartitionedcall_args_2.
*batchnorm_5_statefulpartitionedcall_args_3.
*batchnorm_5_statefulpartitionedcall_args_4+
'conv2d_6_statefulpartitionedcall_args_1+
'conv2d_6_statefulpartitionedcall_args_2.
*batchnorm_6_statefulpartitionedcall_args_1.
*batchnorm_6_statefulpartitionedcall_args_2.
*batchnorm_6_statefulpartitionedcall_args_3.
*batchnorm_6_statefulpartitionedcall_args_4*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2.
*batchnorm_7_statefulpartitionedcall_args_1.
*batchnorm_7_statefulpartitionedcall_args_2.
*batchnorm_7_statefulpartitionedcall_args_3.
*batchnorm_7_statefulpartitionedcall_args_4,
(out_layer_statefulpartitionedcall_args_1,
(out_layer_statefulpartitionedcall_args_2
identity??#batchnorm_1/StatefulPartitionedCall?#batchnorm_2/StatefulPartitionedCall?#batchnorm_3/StatefulPartitionedCall?#batchnorm_4/StatefulPartitionedCall?#batchnorm_5/StatefulPartitionedCall?#batchnorm_6/StatefulPartitionedCall?#batchnorm_7/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?!dropout_3/StatefulPartitionedCall?!dropout_4/StatefulPartitionedCall?!out_layer/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputs'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????00@**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_250862"
 conv2d_1/StatefulPartitionedCall?
#batchnorm_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*batchnorm_1_statefulpartitionedcall_args_1*batchnorm_1_statefulpartitionedcall_args_2*batchnorm_1_statefulpartitionedcall_args_3*batchnorm_1_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????00@**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_1_layer_call_and_return_conditional_losses_262112%
#batchnorm_1/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall,batchnorm_1/StatefulPartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????00@**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_252392"
 conv2d_2/StatefulPartitionedCall?
#batchnorm_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*batchnorm_2_statefulpartitionedcall_args_1*batchnorm_2_statefulpartitionedcall_args_2*batchnorm_2_statefulpartitionedcall_args_3*batchnorm_2_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????00@**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_2_layer_call_and_return_conditional_losses_262932%
#batchnorm_2/StatefulPartitionedCall?
maxpool2d_1/PartitionedCallPartitionedCall,batchnorm_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????@**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_maxpool2d_1_layer_call_and_return_conditional_losses_253852
maxpool2d_1/PartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall$maxpool2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????@**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_263602#
!dropout_1/StatefulPartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_254042"
 conv2d_3/StatefulPartitionedCall?
#batchnorm_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*batchnorm_3_statefulpartitionedcall_args_1*batchnorm_3_statefulpartitionedcall_args_2*batchnorm_3_statefulpartitionedcall_args_3*batchnorm_3_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_3_layer_call_and_return_conditional_losses_264142%
#batchnorm_3/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall,batchnorm_3/StatefulPartitionedCall:output:0'conv2d_4_statefulpartitionedcall_args_1'conv2d_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_255572"
 conv2d_4/StatefulPartitionedCall?
#batchnorm_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*batchnorm_4_statefulpartitionedcall_args_1*batchnorm_4_statefulpartitionedcall_args_2*batchnorm_4_statefulpartitionedcall_args_3*batchnorm_4_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_4_layer_call_and_return_conditional_losses_264962%
#batchnorm_4/StatefulPartitionedCall?
maxpool2d_2/PartitionedCallPartitionedCall,batchnorm_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_maxpool2d_2_layer_call_and_return_conditional_losses_257032
maxpool2d_2/PartitionedCall?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall$maxpool2d_2/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_265632#
!dropout_2/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0'conv2d_5_statefulpartitionedcall_args_1'conv2d_5_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_257222"
 conv2d_5/StatefulPartitionedCall?
#batchnorm_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*batchnorm_5_statefulpartitionedcall_args_1*batchnorm_5_statefulpartitionedcall_args_2*batchnorm_5_statefulpartitionedcall_args_3*batchnorm_5_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_5_layer_call_and_return_conditional_losses_266172%
#batchnorm_5/StatefulPartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall,batchnorm_5/StatefulPartitionedCall:output:0'conv2d_6_statefulpartitionedcall_args_1'conv2d_6_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_258752"
 conv2d_6/StatefulPartitionedCall?
#batchnorm_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*batchnorm_6_statefulpartitionedcall_args_1*batchnorm_6_statefulpartitionedcall_args_2*batchnorm_6_statefulpartitionedcall_args_3*batchnorm_6_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_6_layer_call_and_return_conditional_losses_266992%
#batchnorm_6/StatefulPartitionedCall?
maxpool2d_3/PartitionedCallPartitionedCall,batchnorm_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_maxpool2d_3_layer_call_and_return_conditional_losses_260212
maxpool2d_3/PartitionedCall?
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall$maxpool2d_3/PartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_267662#
!dropout_3/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall*dropout_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????H**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_267902
flatten/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_268092!
dense_1/StatefulPartitionedCall?
#batchnorm_7/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0*batchnorm_7_statefulpartitionedcall_args_1*batchnorm_7_statefulpartitionedcall_args_2*batchnorm_7_statefulpartitionedcall_args_3*batchnorm_7_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_7_layer_call_and_return_conditional_losses_261322%
#batchnorm_7/StatefulPartitionedCall?
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall,batchnorm_7/StatefulPartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_268642#
!dropout_4/StatefulPartitionedCall?
!out_layer/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0(out_layer_statefulpartitionedcall_args_1(out_layer_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_out_layer_layer_call_and_return_conditional_losses_268932#
!out_layer/StatefulPartitionedCall?
IdentityIdentity*out_layer/StatefulPartitionedCall:output:0$^batchnorm_1/StatefulPartitionedCall$^batchnorm_2/StatefulPartitionedCall$^batchnorm_3/StatefulPartitionedCall$^batchnorm_4/StatefulPartitionedCall$^batchnorm_5/StatefulPartitionedCall$^batchnorm_6/StatefulPartitionedCall$^batchnorm_7/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^out_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????00::::::::::::::::::::::::::::::::::::::::::::2J
#batchnorm_1/StatefulPartitionedCall#batchnorm_1/StatefulPartitionedCall2J
#batchnorm_2/StatefulPartitionedCall#batchnorm_2/StatefulPartitionedCall2J
#batchnorm_3/StatefulPartitionedCall#batchnorm_3/StatefulPartitionedCall2J
#batchnorm_4/StatefulPartitionedCall#batchnorm_4/StatefulPartitionedCall2J
#batchnorm_5/StatefulPartitionedCall#batchnorm_5/StatefulPartitionedCall2J
#batchnorm_6/StatefulPartitionedCall#batchnorm_6/StatefulPartitionedCall2J
#batchnorm_7/StatefulPartitionedCall#batchnorm_7/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!out_layer/StatefulPartitionedCall!out_layer/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_27384
conv2d_1_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32#
statefulpartitionedcall_args_33#
statefulpartitionedcall_args_34#
statefulpartitionedcall_args_35#
statefulpartitionedcall_args_36#
statefulpartitionedcall_args_37#
statefulpartitionedcall_args_38#
statefulpartitionedcall_args_39#
statefulpartitionedcall_args_40#
statefulpartitionedcall_args_41#
statefulpartitionedcall_args_42#
statefulpartitionedcall_args_43#
statefulpartitionedcall_args_44
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32statefulpartitionedcall_args_33statefulpartitionedcall_args_34statefulpartitionedcall_args_35statefulpartitionedcall_args_36statefulpartitionedcall_args_37statefulpartitionedcall_args_38statefulpartitionedcall_args_39statefulpartitionedcall_args_40statefulpartitionedcall_args_41statefulpartitionedcall_args_42statefulpartitionedcall_args_43statefulpartitionedcall_args_44*8
Tin1
/2-*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*)
f$R"
 __inference__wrapped_model_250732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????00::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:. *
(
_user_specified_nameconv2d_1_input
?
?
+__inference_batchnorm_4_layer_call_fn_28623

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_4_layer_call_and_return_conditional_losses_256902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_26360

inputs
identity?a
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/random_uniform/max?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniform?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:?????????@2
dropout/random_uniform/mul?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:?????????@2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*/
_output_shapes
:?????????@2
dropout/GreaterEqualx
dropout/mulMulinputsdropout/truediv:z:0*
T0*/
_output_shapes
:?????????@2
dropout/mul?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
dropout/Cast?
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
dropout/mul_1m
IdentityIdentitydropout/mul_1:z:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:& "
 
_user_specified_nameinputs
?
?
$__inference_DCNN_layer_call_fn_27098
conv2d_1_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32#
statefulpartitionedcall_args_33#
statefulpartitionedcall_args_34#
statefulpartitionedcall_args_35#
statefulpartitionedcall_args_36#
statefulpartitionedcall_args_37#
statefulpartitionedcall_args_38#
statefulpartitionedcall_args_39#
statefulpartitionedcall_args_40#
statefulpartitionedcall_args_41#
statefulpartitionedcall_args_42#
statefulpartitionedcall_args_43#
statefulpartitionedcall_args_44
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32statefulpartitionedcall_args_33statefulpartitionedcall_args_34statefulpartitionedcall_args_35statefulpartitionedcall_args_36statefulpartitionedcall_args_37statefulpartitionedcall_args_38statefulpartitionedcall_args_39statefulpartitionedcall_args_40statefulpartitionedcall_args_41statefulpartitionedcall_args_42statefulpartitionedcall_args_43statefulpartitionedcall_args_44*8
Tin1
/2-*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_DCNN_layer_call_and_return_conditional_losses_270512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????00::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:. *
(
_user_specified_nameconv2d_1_input
?#
?
F__inference_batchnorm_3_layer_call_and_return_conditional_losses_26414

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_26399
assignmovingavg_1_26406
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *?p}?2	
Const_2?
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/26399*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/26399*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_26399*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/26399*
_output_shapes	
:?2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/26399*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_26399AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/26399*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/26406*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/26406*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_26406*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/26406*
_output_shapes	
:?2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/26406*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_26406AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/26406*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?
b
)__inference_dropout_4_layer_call_fn_29262

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_268642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
G
+__inference_maxpool2d_2_layer_call_fn_25709

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4????????????????????????????????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_maxpool2d_2_layer_call_and_return_conditional_losses_257032
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?
?
(__inference_conv2d_3_layer_call_fn_25412

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,????????????????????????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_254042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
+__inference_batchnorm_2_layer_call_fn_28342

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????00@**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_batchnorm_2_layer_call_and_return_conditional_losses_263152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????00@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????00@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?#
?
F__inference_batchnorm_6_layer_call_and_return_conditional_losses_29012

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_28997
assignmovingavg_1_29004
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *?p}?2	
Const_2?
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/28997*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/28997*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_28997*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/28997*
_output_shapes	
:?2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/28997*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_28997AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/28997*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/29004*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/29004*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_29004*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/29004*
_output_shapes	
:?2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/29004*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_29004AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/29004*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
?
b
)__inference_dropout_2_layer_call_fn_28727

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_265632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
F__inference_batchnorm_6_layer_call_and_return_conditional_losses_29034

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
Const?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
Q
conv2d_1_input?
 serving_default_conv2d_1_input:0?????????00=
	out_layer0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
ѧ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer-11
layer-12
layer_with_weights-8
layer-13
layer_with_weights-9
layer-14
layer_with_weights-10
layer-15
layer_with_weights-11
layer-16
layer-17
layer-18
layer-19
layer_with_weights-12
layer-20
layer_with_weights-13
layer-21
layer-22
layer_with_weights-14
layer-23
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"??
_tf_keras_sequential??{"class_name": "Sequential", "name": "DCNN", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "DCNN", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "batch_input_shape": [null, 48, 48, 1], "dtype": "float32", "filters": 64, "kernel_size": [5, 5], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batchnorm_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [5, 5], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batchnorm_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batchnorm_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batchnorm_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool2d_2", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batchnorm_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batchnorm_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool2d_3", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batchnorm_7", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.6, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "out_layer", "trainable": true, "dtype": "float32", "units": 7, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "DCNN", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "batch_input_shape": [null, 48, 48, 1], "dtype": "float32", "filters": 64, "kernel_size": [5, 5], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batchnorm_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [5, 5], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batchnorm_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batchnorm_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batchnorm_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool2d_2", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batchnorm_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batchnorm_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool2d_3", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batchnorm_7", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.6, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "out_layer", "trainable": true, "dtype": "float32", "units": 7, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0001250000059371814, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "conv2d_1_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 48, 48, 1], "config": {"batch_input_shape": [null, 48, 48, 1], "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_1_input"}}
?

kernel
 bias
!regularization_losses
"trainable_variables
#	variables
$	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 48, 48, 1], "config": {"name": "conv2d_1", "trainable": true, "batch_input_shape": [null, 48, 48, 1], "dtype": "float32", "filters": 64, "kernel_size": [5, 5], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}}
?
%axis
	&gamma
'beta
(moving_mean
)moving_variance
*regularization_losses
+trainable_variables
,	variables
-	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batchnorm_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batchnorm_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}}
?

.kernel
/bias
0regularization_losses
1trainable_variables
2	variables
3	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [5, 5], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
?
4axis
	5gamma
6beta
7moving_mean
8moving_variance
9regularization_losses
:trainable_variables
;	variables
<	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batchnorm_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batchnorm_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}}
?
=regularization_losses
>trainable_variables
?	variables
@	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "maxpool2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "maxpool2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}
?

Ekernel
Fbias
Gregularization_losses
Htrainable_variables
I	variables
J	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
?
Kaxis
	Lgamma
Mbeta
Nmoving_mean
Omoving_variance
Pregularization_losses
Qtrainable_variables
R	variables
S	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batchnorm_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batchnorm_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}}
?

Tkernel
Ubias
Vregularization_losses
Wtrainable_variables
X	variables
Y	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}}
?
Zaxis
	[gamma
\beta
]moving_mean
^moving_variance
_regularization_losses
`trainable_variables
a	variables
b	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batchnorm_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batchnorm_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}}
?
cregularization_losses
dtrainable_variables
e	variables
f	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "maxpool2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "maxpool2d_2", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
gregularization_losses
htrainable_variables
i	variables
j	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}
?

kkernel
lbias
mregularization_losses
ntrainable_variables
o	variables
p	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}}
?
qaxis
	rgamma
sbeta
tmoving_mean
umoving_variance
vregularization_losses
wtrainable_variables
x	variables
y	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batchnorm_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batchnorm_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}}}
?

zkernel
{bias
|regularization_losses
}trainable_variables
~	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}}
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batchnorm_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batchnorm_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "maxpool2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "maxpool2d_3", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 9216}}}}
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batchnorm_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batchnorm_7", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 128}}}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.6, "noise_shape": null, "seed": null}}
?
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "out_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "out_layer", "trainable": true, "dtype": "float32", "units": 7, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
?
	?iter
?beta_1
?beta_2

?decay
?learning_ratem? m?&m?'m?.m?/m?5m?6m?Em?Fm?Lm?Mm?Tm?Um?[m?\m?km?lm?rm?sm?zm?{m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?v? v?&v?'v?.v?/v?5v?6v?Ev?Fv?Lv?Mv?Tv?Uv?[v?\v?kv?lv?rv?sv?zv?{v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?"
	optimizer
 "
trackable_list_wrapper
?
0
 1
&2
'3
.4
/5
56
67
E8
F9
L10
M11
T12
U13
[14
\15
k16
l17
r18
s19
z20
{21
?22
?23
?24
?25
?26
?27
?28
?29"
trackable_list_wrapper
?
0
 1
&2
'3
(4
)5
.6
/7
58
69
710
811
E12
F13
L14
M15
N16
O17
T18
U19
[20
\21
]22
^23
k24
l25
r26
s27
t28
u29
z30
{31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43"
trackable_list_wrapper
?
?layers
regularization_losses
?metrics
trainable_variables
	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
+:)@2conv2d_1_1/kernel
:@2conv2d_1_1/bias
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
?
?layers
!regularization_losses
?metrics
"trainable_variables
#	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
!:@2batchnorm_1_1/gamma
 :@2batchnorm_1_1/beta
):'@ (2batchnorm_1_1/moving_mean
-:+@ (2batchnorm_1_1/moving_variance
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
<
&0
'1
(2
)3"
trackable_list_wrapper
?
?layers
*regularization_losses
?metrics
+trainable_variables
,	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)@@2conv2d_2_1/kernel
:@2conv2d_2_1/bias
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
?
?layers
0regularization_losses
?metrics
1trainable_variables
2	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
!:@2batchnorm_2_1/gamma
 :@2batchnorm_2_1/beta
):'@ (2batchnorm_2_1/moving_mean
-:+@ (2batchnorm_2_1/moving_variance
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
<
50
61
72
83"
trackable_list_wrapper
?
?layers
9regularization_losses
?metrics
:trainable_variables
;	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
=regularization_losses
?metrics
>trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
Aregularization_losses
?metrics
Btrainable_variables
C	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*@?2conv2d_3_1/kernel
:?2conv2d_3_1/bias
 "
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
?
?layers
Gregularization_losses
?metrics
Htrainable_variables
I	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
": ?2batchnorm_3_1/gamma
!:?2batchnorm_3_1/beta
*:(? (2batchnorm_3_1/moving_mean
.:,? (2batchnorm_3_1/moving_variance
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
<
L0
M1
N2
O3"
trackable_list_wrapper
?
?layers
Pregularization_losses
?metrics
Qtrainable_variables
R	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-:+??2conv2d_4_1/kernel
:?2conv2d_4_1/bias
 "
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
?
?layers
Vregularization_losses
?metrics
Wtrainable_variables
X	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
": ?2batchnorm_4_1/gamma
!:?2batchnorm_4_1/beta
*:(? (2batchnorm_4_1/moving_mean
.:,? (2batchnorm_4_1/moving_variance
 "
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
<
[0
\1
]2
^3"
trackable_list_wrapper
?
?layers
_regularization_losses
?metrics
`trainable_variables
a	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
cregularization_losses
?metrics
dtrainable_variables
e	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
gregularization_losses
?metrics
htrainable_variables
i	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-:+??2conv2d_5_1/kernel
:?2conv2d_5_1/bias
 "
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
?
?layers
mregularization_losses
?metrics
ntrainable_variables
o	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
": ?2batchnorm_5_1/gamma
!:?2batchnorm_5_1/beta
*:(? (2batchnorm_5_1/moving_mean
.:,? (2batchnorm_5_1/moving_variance
 "
trackable_list_wrapper
.
r0
s1"
trackable_list_wrapper
<
r0
s1
t2
u3"
trackable_list_wrapper
?
?layers
vregularization_losses
?metrics
wtrainable_variables
x	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-:+??2conv2d_6_1/kernel
:?2conv2d_6_1/bias
 "
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
?
?layers
|regularization_losses
?metrics
}trainable_variables
~	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
": ?2batchnorm_6_1/gamma
!:?2batchnorm_6_1/beta
*:(? (2batchnorm_6_1/moving_mean
.:,? (2batchnorm_6_1/moving_variance
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?layers
?regularization_losses
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?regularization_losses
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?regularization_losses
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?regularization_losses
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
$:"
?H?2dense_1_1/kernel
:?2dense_1_1/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layers
?regularization_losses
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
": ?2batchnorm_7_1/gamma
!:?2batchnorm_7_1/beta
*:(? (2batchnorm_7_1/moving_mean
.:,? (2batchnorm_7_1/moving_variance
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?layers
?regularization_losses
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?regularization_losses
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#	?2out_layer_1/kernel
:2out_layer_1/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layers
?regularization_losses
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?
0
1
2
3
4
5
6
	7

8
9
10
11
12
13
14
15
16
17
18
19
20
21
22"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
(0
)1
72
83
N4
O5
]6
^7
t8
u9
?10
?11
?12
?13"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?
_fn_kwargs
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layers
?regularization_losses
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0:.@2Adam/conv2d_1_1/kernel/m
": @2Adam/conv2d_1_1/bias/m
&:$@2Adam/batchnorm_1_1/gamma/m
%:#@2Adam/batchnorm_1_1/beta/m
0:.@@2Adam/conv2d_2_1/kernel/m
": @2Adam/conv2d_2_1/bias/m
&:$@2Adam/batchnorm_2_1/gamma/m
%:#@2Adam/batchnorm_2_1/beta/m
1:/@?2Adam/conv2d_3_1/kernel/m
#:!?2Adam/conv2d_3_1/bias/m
':%?2Adam/batchnorm_3_1/gamma/m
&:$?2Adam/batchnorm_3_1/beta/m
2:0??2Adam/conv2d_4_1/kernel/m
#:!?2Adam/conv2d_4_1/bias/m
':%?2Adam/batchnorm_4_1/gamma/m
&:$?2Adam/batchnorm_4_1/beta/m
2:0??2Adam/conv2d_5_1/kernel/m
#:!?2Adam/conv2d_5_1/bias/m
':%?2Adam/batchnorm_5_1/gamma/m
&:$?2Adam/batchnorm_5_1/beta/m
2:0??2Adam/conv2d_6_1/kernel/m
#:!?2Adam/conv2d_6_1/bias/m
':%?2Adam/batchnorm_6_1/gamma/m
&:$?2Adam/batchnorm_6_1/beta/m
):'
?H?2Adam/dense_1_1/kernel/m
": ?2Adam/dense_1_1/bias/m
':%?2Adam/batchnorm_7_1/gamma/m
&:$?2Adam/batchnorm_7_1/beta/m
*:(	?2Adam/out_layer_1/kernel/m
#:!2Adam/out_layer_1/bias/m
0:.@2Adam/conv2d_1_1/kernel/v
": @2Adam/conv2d_1_1/bias/v
&:$@2Adam/batchnorm_1_1/gamma/v
%:#@2Adam/batchnorm_1_1/beta/v
0:.@@2Adam/conv2d_2_1/kernel/v
": @2Adam/conv2d_2_1/bias/v
&:$@2Adam/batchnorm_2_1/gamma/v
%:#@2Adam/batchnorm_2_1/beta/v
1:/@?2Adam/conv2d_3_1/kernel/v
#:!?2Adam/conv2d_3_1/bias/v
':%?2Adam/batchnorm_3_1/gamma/v
&:$?2Adam/batchnorm_3_1/beta/v
2:0??2Adam/conv2d_4_1/kernel/v
#:!?2Adam/conv2d_4_1/bias/v
':%?2Adam/batchnorm_4_1/gamma/v
&:$?2Adam/batchnorm_4_1/beta/v
2:0??2Adam/conv2d_5_1/kernel/v
#:!?2Adam/conv2d_5_1/bias/v
':%?2Adam/batchnorm_5_1/gamma/v
&:$?2Adam/batchnorm_5_1/beta/v
2:0??2Adam/conv2d_6_1/kernel/v
#:!?2Adam/conv2d_6_1/bias/v
':%?2Adam/batchnorm_6_1/gamma/v
&:$?2Adam/batchnorm_6_1/beta/v
):'
?H?2Adam/dense_1_1/kernel/v
": ?2Adam/dense_1_1/bias/v
':%?2Adam/batchnorm_7_1/gamma/v
&:$?2Adam/batchnorm_7_1/beta/v
*:(	?2Adam/out_layer_1/kernel/v
#:!2Adam/out_layer_1/bias/v
?2?
$__inference_DCNN_layer_call_fn_27218
$__inference_DCNN_layer_call_fn_28022
$__inference_DCNN_layer_call_fn_27098
$__inference_DCNN_layer_call_fn_27973?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_25073?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *5?2
0?-
conv2d_1_input?????????00
?2?
?__inference_DCNN_layer_call_and_return_conditional_losses_27924
?__inference_DCNN_layer_call_and_return_conditional_losses_26977
?__inference_DCNN_layer_call_and_return_conditional_losses_27728
?__inference_DCNN_layer_call_and_return_conditional_losses_26906?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_conv2d_1_layer_call_fn_25094?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_25086?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
+__inference_batchnorm_1_layer_call_fn_28108
+__inference_batchnorm_1_layer_call_fn_28182
+__inference_batchnorm_1_layer_call_fn_28099
+__inference_batchnorm_1_layer_call_fn_28173?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_batchnorm_1_layer_call_and_return_conditional_losses_28068
F__inference_batchnorm_1_layer_call_and_return_conditional_losses_28164
F__inference_batchnorm_1_layer_call_and_return_conditional_losses_28090
F__inference_batchnorm_1_layer_call_and_return_conditional_losses_28142?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_conv2d_2_layer_call_fn_25247?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_25239?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
+__inference_batchnorm_2_layer_call_fn_28268
+__inference_batchnorm_2_layer_call_fn_28333
+__inference_batchnorm_2_layer_call_fn_28259
+__inference_batchnorm_2_layer_call_fn_28342?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_batchnorm_2_layer_call_and_return_conditional_losses_28228
F__inference_batchnorm_2_layer_call_and_return_conditional_losses_28250
F__inference_batchnorm_2_layer_call_and_return_conditional_losses_28302
F__inference_batchnorm_2_layer_call_and_return_conditional_losses_28324?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_maxpool2d_1_layer_call_fn_25391?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
F__inference_maxpool2d_1_layer_call_and_return_conditional_losses_25385?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
)__inference_dropout_1_layer_call_fn_28377
)__inference_dropout_1_layer_call_fn_28372?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_dropout_1_layer_call_and_return_conditional_losses_28362
D__inference_dropout_1_layer_call_and_return_conditional_losses_28367?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_conv2d_3_layer_call_fn_25412?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_25404?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
+__inference_batchnorm_3_layer_call_fn_28537
+__inference_batchnorm_3_layer_call_fn_28454
+__inference_batchnorm_3_layer_call_fn_28463
+__inference_batchnorm_3_layer_call_fn_28528?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_batchnorm_3_layer_call_and_return_conditional_losses_28423
F__inference_batchnorm_3_layer_call_and_return_conditional_losses_28445
F__inference_batchnorm_3_layer_call_and_return_conditional_losses_28497
F__inference_batchnorm_3_layer_call_and_return_conditional_losses_28519?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_conv2d_4_layer_call_fn_25565?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_25557?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
+__inference_batchnorm_4_layer_call_fn_28623
+__inference_batchnorm_4_layer_call_fn_28614
+__inference_batchnorm_4_layer_call_fn_28688
+__inference_batchnorm_4_layer_call_fn_28697?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_batchnorm_4_layer_call_and_return_conditional_losses_28605
F__inference_batchnorm_4_layer_call_and_return_conditional_losses_28583
F__inference_batchnorm_4_layer_call_and_return_conditional_losses_28657
F__inference_batchnorm_4_layer_call_and_return_conditional_losses_28679?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_maxpool2d_2_layer_call_fn_25709?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
F__inference_maxpool2d_2_layer_call_and_return_conditional_losses_25703?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
)__inference_dropout_2_layer_call_fn_28727
)__inference_dropout_2_layer_call_fn_28732?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_dropout_2_layer_call_and_return_conditional_losses_28717
D__inference_dropout_2_layer_call_and_return_conditional_losses_28722?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_conv2d_5_layer_call_fn_25730?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
C__inference_conv2d_5_layer_call_and_return_conditional_losses_25722?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
+__inference_batchnorm_5_layer_call_fn_28892
+__inference_batchnorm_5_layer_call_fn_28883
+__inference_batchnorm_5_layer_call_fn_28818
+__inference_batchnorm_5_layer_call_fn_28809?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_batchnorm_5_layer_call_and_return_conditional_losses_28852
F__inference_batchnorm_5_layer_call_and_return_conditional_losses_28874
F__inference_batchnorm_5_layer_call_and_return_conditional_losses_28800
F__inference_batchnorm_5_layer_call_and_return_conditional_losses_28778?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_conv2d_6_layer_call_fn_25883?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
C__inference_conv2d_6_layer_call_and_return_conditional_losses_25875?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
+__inference_batchnorm_6_layer_call_fn_28969
+__inference_batchnorm_6_layer_call_fn_29052
+__inference_batchnorm_6_layer_call_fn_29043
+__inference_batchnorm_6_layer_call_fn_28978?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_batchnorm_6_layer_call_and_return_conditional_losses_29034
F__inference_batchnorm_6_layer_call_and_return_conditional_losses_28960
F__inference_batchnorm_6_layer_call_and_return_conditional_losses_29012
F__inference_batchnorm_6_layer_call_and_return_conditional_losses_28938?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_maxpool2d_3_layer_call_fn_26027?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
F__inference_maxpool2d_3_layer_call_and_return_conditional_losses_26021?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
)__inference_dropout_3_layer_call_fn_29087
)__inference_dropout_3_layer_call_fn_29082?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_dropout_3_layer_call_and_return_conditional_losses_29072
D__inference_dropout_3_layer_call_and_return_conditional_losses_29077?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_flatten_layer_call_fn_29098?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_flatten_layer_call_and_return_conditional_losses_29093?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_1_layer_call_fn_29116?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_1_layer_call_and_return_conditional_losses_29109?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_batchnorm_7_layer_call_fn_29232
+__inference_batchnorm_7_layer_call_fn_29223?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_batchnorm_7_layer_call_and_return_conditional_losses_29191
F__inference_batchnorm_7_layer_call_and_return_conditional_losses_29214?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dropout_4_layer_call_fn_29262
)__inference_dropout_4_layer_call_fn_29267?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_dropout_4_layer_call_and_return_conditional_losses_29252
D__inference_dropout_4_layer_call_and_return_conditional_losses_29257?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_out_layer_layer_call_fn_29285?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_out_layer_layer_call_and_return_conditional_losses_29278?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
9B7
#__inference_signature_wrapper_27384conv2d_1_input
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 ?
?__inference_DCNN_layer_call_and_return_conditional_losses_26906?8 &'()./5678EFLMNOTU[\]^klrstuz{????????????G?D
=?:
0?-
conv2d_1_input?????????00
p

 
? "%?"
?
0?????????
? ?
?__inference_DCNN_layer_call_and_return_conditional_losses_26977?8 &'()./5678EFLMNOTU[\]^klrstuz{????????????G?D
=?:
0?-
conv2d_1_input?????????00
p 

 
? "%?"
?
0?????????
? ?
?__inference_DCNN_layer_call_and_return_conditional_losses_27728?8 &'()./5678EFLMNOTU[\]^klrstuz{??????????????<
5?2
(?%
inputs?????????00
p

 
? "%?"
?
0?????????
? ?
?__inference_DCNN_layer_call_and_return_conditional_losses_27924?8 &'()./5678EFLMNOTU[\]^klrstuz{??????????????<
5?2
(?%
inputs?????????00
p 

 
? "%?"
?
0?????????
? ?
$__inference_DCNN_layer_call_fn_27098?8 &'()./5678EFLMNOTU[\]^klrstuz{????????????G?D
=?:
0?-
conv2d_1_input?????????00
p

 
? "???????????
$__inference_DCNN_layer_call_fn_27218?8 &'()./5678EFLMNOTU[\]^klrstuz{????????????G?D
=?:
0?-
conv2d_1_input?????????00
p 

 
? "???????????
$__inference_DCNN_layer_call_fn_27973?8 &'()./5678EFLMNOTU[\]^klrstuz{??????????????<
5?2
(?%
inputs?????????00
p

 
? "???????????
$__inference_DCNN_layer_call_fn_28022?8 &'()./5678EFLMNOTU[\]^klrstuz{??????????????<
5?2
(?%
inputs?????????00
p 

 
? "???????????
 __inference__wrapped_model_25073?8 &'()./5678EFLMNOTU[\]^klrstuz{??????????????<
5?2
0?-
conv2d_1_input?????????00
? "5?2
0
	out_layer#? 
	out_layer??????????
F__inference_batchnorm_1_layer_call_and_return_conditional_losses_28068?&'()M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
F__inference_batchnorm_1_layer_call_and_return_conditional_losses_28090?&'()M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
F__inference_batchnorm_1_layer_call_and_return_conditional_losses_28142r&'();?8
1?.
(?%
inputs?????????00@
p
? "-?*
#? 
0?????????00@
? ?
F__inference_batchnorm_1_layer_call_and_return_conditional_losses_28164r&'();?8
1?.
(?%
inputs?????????00@
p 
? "-?*
#? 
0?????????00@
? ?
+__inference_batchnorm_1_layer_call_fn_28099?&'()M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
+__inference_batchnorm_1_layer_call_fn_28108?&'()M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
+__inference_batchnorm_1_layer_call_fn_28173e&'();?8
1?.
(?%
inputs?????????00@
p
? " ??????????00@?
+__inference_batchnorm_1_layer_call_fn_28182e&'();?8
1?.
(?%
inputs?????????00@
p 
? " ??????????00@?
F__inference_batchnorm_2_layer_call_and_return_conditional_losses_28228?5678M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
F__inference_batchnorm_2_layer_call_and_return_conditional_losses_28250?5678M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
F__inference_batchnorm_2_layer_call_and_return_conditional_losses_28302r5678;?8
1?.
(?%
inputs?????????00@
p
? "-?*
#? 
0?????????00@
? ?
F__inference_batchnorm_2_layer_call_and_return_conditional_losses_28324r5678;?8
1?.
(?%
inputs?????????00@
p 
? "-?*
#? 
0?????????00@
? ?
+__inference_batchnorm_2_layer_call_fn_28259?5678M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
+__inference_batchnorm_2_layer_call_fn_28268?5678M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
+__inference_batchnorm_2_layer_call_fn_28333e5678;?8
1?.
(?%
inputs?????????00@
p
? " ??????????00@?
+__inference_batchnorm_2_layer_call_fn_28342e5678;?8
1?.
(?%
inputs?????????00@
p 
? " ??????????00@?
F__inference_batchnorm_3_layer_call_and_return_conditional_losses_28423?LMNON?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
F__inference_batchnorm_3_layer_call_and_return_conditional_losses_28445?LMNON?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
F__inference_batchnorm_3_layer_call_and_return_conditional_losses_28497tLMNO<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
F__inference_batchnorm_3_layer_call_and_return_conditional_losses_28519tLMNO<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
+__inference_batchnorm_3_layer_call_fn_28454?LMNON?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
+__inference_batchnorm_3_layer_call_fn_28463?LMNON?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
+__inference_batchnorm_3_layer_call_fn_28528gLMNO<?9
2?/
)?&
inputs??????????
p
? "!????????????
+__inference_batchnorm_3_layer_call_fn_28537gLMNO<?9
2?/
)?&
inputs??????????
p 
? "!????????????
F__inference_batchnorm_4_layer_call_and_return_conditional_losses_28583?[\]^N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
F__inference_batchnorm_4_layer_call_and_return_conditional_losses_28605?[\]^N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
F__inference_batchnorm_4_layer_call_and_return_conditional_losses_28657t[\]^<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
F__inference_batchnorm_4_layer_call_and_return_conditional_losses_28679t[\]^<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
+__inference_batchnorm_4_layer_call_fn_28614?[\]^N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
+__inference_batchnorm_4_layer_call_fn_28623?[\]^N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
+__inference_batchnorm_4_layer_call_fn_28688g[\]^<?9
2?/
)?&
inputs??????????
p
? "!????????????
+__inference_batchnorm_4_layer_call_fn_28697g[\]^<?9
2?/
)?&
inputs??????????
p 
? "!????????????
F__inference_batchnorm_5_layer_call_and_return_conditional_losses_28778trstu<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
F__inference_batchnorm_5_layer_call_and_return_conditional_losses_28800trstu<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
F__inference_batchnorm_5_layer_call_and_return_conditional_losses_28852?rstuN?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
F__inference_batchnorm_5_layer_call_and_return_conditional_losses_28874?rstuN?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
+__inference_batchnorm_5_layer_call_fn_28809grstu<?9
2?/
)?&
inputs??????????
p
? "!????????????
+__inference_batchnorm_5_layer_call_fn_28818grstu<?9
2?/
)?&
inputs??????????
p 
? "!????????????
+__inference_batchnorm_5_layer_call_fn_28883?rstuN?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
+__inference_batchnorm_5_layer_call_fn_28892?rstuN?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
F__inference_batchnorm_6_layer_call_and_return_conditional_losses_28938?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
F__inference_batchnorm_6_layer_call_and_return_conditional_losses_28960?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
F__inference_batchnorm_6_layer_call_and_return_conditional_losses_29012x????<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
F__inference_batchnorm_6_layer_call_and_return_conditional_losses_29034x????<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
+__inference_batchnorm_6_layer_call_fn_28969?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
+__inference_batchnorm_6_layer_call_fn_28978?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
+__inference_batchnorm_6_layer_call_fn_29043k????<?9
2?/
)?&
inputs??????????
p
? "!????????????
+__inference_batchnorm_6_layer_call_fn_29052k????<?9
2?/
)?&
inputs??????????
p 
? "!????????????
F__inference_batchnorm_7_layer_call_and_return_conditional_losses_29191h????4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
F__inference_batchnorm_7_layer_call_and_return_conditional_losses_29214h????4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
+__inference_batchnorm_7_layer_call_fn_29223[????4?1
*?'
!?
inputs??????????
p
? "????????????
+__inference_batchnorm_7_layer_call_fn_29232[????4?1
*?'
!?
inputs??????????
p 
? "????????????
C__inference_conv2d_1_layer_call_and_return_conditional_losses_25086? I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????@
? ?
(__inference_conv2d_1_layer_call_fn_25094? I?F
??<
:?7
inputs+???????????????????????????
? "2?/+???????????????????????????@?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_25239?./I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????@
? ?
(__inference_conv2d_2_layer_call_fn_25247?./I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+???????????????????????????@?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_25404?EFI?F
??<
:?7
inputs+???????????????????????????@
? "@?=
6?3
0,????????????????????????????
? ?
(__inference_conv2d_3_layer_call_fn_25412?EFI?F
??<
:?7
inputs+???????????????????????????@
? "3?0,?????????????????????????????
C__inference_conv2d_4_layer_call_and_return_conditional_losses_25557?TUJ?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
(__inference_conv2d_4_layer_call_fn_25565?TUJ?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
C__inference_conv2d_5_layer_call_and_return_conditional_losses_25722?klJ?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
(__inference_conv2d_5_layer_call_fn_25730?klJ?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
C__inference_conv2d_6_layer_call_and_return_conditional_losses_25875?z{J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
(__inference_conv2d_6_layer_call_fn_25883?z{J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
B__inference_dense_1_layer_call_and_return_conditional_losses_29109`??0?-
&?#
!?
inputs??????????H
? "&?#
?
0??????????
? ~
'__inference_dense_1_layer_call_fn_29116S??0?-
&?#
!?
inputs??????????H
? "????????????
D__inference_dropout_1_layer_call_and_return_conditional_losses_28362l;?8
1?.
(?%
inputs?????????@
p
? "-?*
#? 
0?????????@
? ?
D__inference_dropout_1_layer_call_and_return_conditional_losses_28367l;?8
1?.
(?%
inputs?????????@
p 
? "-?*
#? 
0?????????@
? ?
)__inference_dropout_1_layer_call_fn_28372_;?8
1?.
(?%
inputs?????????@
p
? " ??????????@?
)__inference_dropout_1_layer_call_fn_28377_;?8
1?.
(?%
inputs?????????@
p 
? " ??????????@?
D__inference_dropout_2_layer_call_and_return_conditional_losses_28717n<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
D__inference_dropout_2_layer_call_and_return_conditional_losses_28722n<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
)__inference_dropout_2_layer_call_fn_28727a<?9
2?/
)?&
inputs??????????
p
? "!????????????
)__inference_dropout_2_layer_call_fn_28732a<?9
2?/
)?&
inputs??????????
p 
? "!????????????
D__inference_dropout_3_layer_call_and_return_conditional_losses_29072n<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
D__inference_dropout_3_layer_call_and_return_conditional_losses_29077n<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
)__inference_dropout_3_layer_call_fn_29082a<?9
2?/
)?&
inputs??????????
p
? "!????????????
)__inference_dropout_3_layer_call_fn_29087a<?9
2?/
)?&
inputs??????????
p 
? "!????????????
D__inference_dropout_4_layer_call_and_return_conditional_losses_29252^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
D__inference_dropout_4_layer_call_and_return_conditional_losses_29257^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ~
)__inference_dropout_4_layer_call_fn_29262Q4?1
*?'
!?
inputs??????????
p
? "???????????~
)__inference_dropout_4_layer_call_fn_29267Q4?1
*?'
!?
inputs??????????
p 
? "????????????
B__inference_flatten_layer_call_and_return_conditional_losses_29093b8?5
.?+
)?&
inputs??????????
? "&?#
?
0??????????H
? ?
'__inference_flatten_layer_call_fn_29098U8?5
.?+
)?&
inputs??????????
? "???????????H?
F__inference_maxpool2d_1_layer_call_and_return_conditional_losses_25385?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
+__inference_maxpool2d_1_layer_call_fn_25391?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
F__inference_maxpool2d_2_layer_call_and_return_conditional_losses_25703?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
+__inference_maxpool2d_2_layer_call_fn_25709?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
F__inference_maxpool2d_3_layer_call_and_return_conditional_losses_26021?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
+__inference_maxpool2d_3_layer_call_fn_26027?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
D__inference_out_layer_layer_call_and_return_conditional_losses_29278_??0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? 
)__inference_out_layer_layer_call_fn_29285R??0?-
&?#
!?
inputs??????????
? "???????????
#__inference_signature_wrapper_27384?8 &'()./5678EFLMNOTU[\]^klrstuz{????????????Q?N
? 
G?D
B
conv2d_1_input0?-
conv2d_1_input?????????00"5?2
0
	out_layer#? 
	out_layer?????????