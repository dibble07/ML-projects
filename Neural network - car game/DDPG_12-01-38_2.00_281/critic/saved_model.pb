��
��
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
dtypetype�
�
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
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8��
|
critic/L1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:0@*!
shared_namecritic/L1/kernel
u
$critic/L1/kernel/Read/ReadVariableOpReadVariableOpcritic/L1/kernel*
_output_shapes

:0@*
dtype0
t
critic/L1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namecritic/L1/bias
m
"critic/L1/bias/Read/ReadVariableOpReadVariableOpcritic/L1/bias*
_output_shapes
:@*
dtype0
|
critic/L2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *!
shared_namecritic/L2/kernel
u
$critic/L2/kernel/Read/ReadVariableOpReadVariableOpcritic/L2/kernel*
_output_shapes

:@ *
dtype0
t
critic/L2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecritic/L2/bias
m
"critic/L2/bias/Read/ReadVariableOpReadVariableOpcritic/L2/bias*
_output_shapes
: *
dtype0
|
critic/L3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *!
shared_namecritic/L3/kernel
u
$critic/L3/kernel/Read/ReadVariableOpReadVariableOpcritic/L3/kernel*
_output_shapes

: *
dtype0
t
critic/L3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namecritic/L3/bias
m
"critic/L3/bias/Read/ReadVariableOpReadVariableOpcritic/L3/bias*
_output_shapes
:*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
z
l1
l2
l3

signatures
trainable_variables
	variables
regularization_losses
	keras_api
h

	kernel

bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
 
*
	0

1
2
3
4
5
*
	0

1
2
3
4
5
 
�
trainable_variables
	variables
layer_regularization_losses
layer_metrics

layers
metrics
non_trainable_variables
regularization_losses
JH
VARIABLE_VALUEcritic/L1/kernel$l1/kernel/.ATTRIBUTES/VARIABLE_VALUE
FD
VARIABLE_VALUEcritic/L1/bias"l1/bias/.ATTRIBUTES/VARIABLE_VALUE

	0

1

	0

1
 
�
trainable_variables
	variables
 layer_regularization_losses
!layer_metrics

"layers
#metrics
$non_trainable_variables
regularization_losses
JH
VARIABLE_VALUEcritic/L2/kernel$l2/kernel/.ATTRIBUTES/VARIABLE_VALUE
FD
VARIABLE_VALUEcritic/L2/bias"l2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
trainable_variables
	variables
%layer_regularization_losses
&layer_metrics

'layers
(metrics
)non_trainable_variables
regularization_losses
JH
VARIABLE_VALUEcritic/L3/kernel$l3/kernel/.ATTRIBUTES/VARIABLE_VALUE
FD
VARIABLE_VALUEcritic/L3/bias"l3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
trainable_variables
	variables
*layer_regularization_losses
+layer_metrics

,layers
-metrics
.non_trainable_variables
regularization_losses
 
 

0
1
2
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
z
serving_default_input_1Placeholder*'
_output_shapes
:���������.*
dtype0*
shape:���������.
z
serving_default_input_2Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2critic/L1/kernelcritic/L1/biascritic/L2/kernelcritic/L2/biascritic/L3/kernelcritic/L3/bias*
Tin

2*
Tout
2*'
_output_shapes
:���������*(
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8*+
f&R$
"__inference_signature_wrapper_4041
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$critic/L1/kernel/Read/ReadVariableOp"critic/L1/bias/Read/ReadVariableOp$critic/L2/kernel/Read/ReadVariableOp"critic/L2/bias/Read/ReadVariableOp$critic/L3/kernel/Read/ReadVariableOp"critic/L3/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*&
f!R
__inference__traced_save_4087
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamecritic/L1/kernelcritic/L1/biascritic/L2/kernelcritic/L2/biascritic/L3/kernelcritic/L3/bias*
Tin
	2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*)
f$R"
 __inference__traced_restore_4117��
�
�
;__inference_L2_layer_call_and_return_conditional_losses_851

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:::O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�!
�
 __inference__traced_restore_4117
file_prefix%
!assignvariableop_critic_l1_kernel%
!assignvariableop_1_critic_l1_bias'
#assignvariableop_2_critic_l2_kernel%
!assignvariableop_3_critic_l2_bias'
#assignvariableop_4_critic_l3_kernel%
!assignvariableop_5_critic_l3_bias

identity_7��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B$l1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"l1/bias/.ATTRIBUTES/VARIABLE_VALUEB$l2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"l2/bias/.ATTRIBUTES/VARIABLE_VALUEB$l3/kernel/.ATTRIBUTES/VARIABLE_VALUEB"l3/bias/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*,
_output_shapes
::::::*
dtypes

22
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp!assignvariableop_critic_l1_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_critic_l1_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp#assignvariableop_2_critic_l2_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp!assignvariableop_3_critic_l2_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp#assignvariableop_4_critic_l3_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp!assignvariableop_5_critic_l3_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5�
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names�
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices�
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
NoOp�

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_6�

Identity_7IdentityIdentity_6:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2

Identity_7"!

identity_7Identity_7:output:0*-
_input_shapes
: ::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�	
�
:__forward_L3_layer_call_and_return_conditional_losses_1717
inputs_0"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
matmul_readvariableop

inputs��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0"
inputsinputs_0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0*.
_input_shapes
:��������� ::*h
backward_function_nameNL__inference___backward_L3_layer_call_and_return_conditional_losses_1705_1718:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�

�
__inference__wrapped_model_4021
input_1
input_2
critic_4007
critic_4009
critic_4011
critic_4013
critic_4015
critic_4017
identity��critic/StatefulPartitionedCall�
critic/StatefulPartitionedCallStatefulPartitionedCallinput_1input_2critic_4007critic_4009critic_4011critic_4013critic_4015critic_4017*
Tin

2*
Tout
2*'
_output_shapes
:���������*(
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8*0
f+R)
'__inference_restored_function_body_15752 
critic/StatefulPartitionedCall�
IdentityIdentity'critic/StatefulPartitionedCall:output:0^critic/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:���������.:���������::::::2@
critic/StatefulPartitionedCallcritic/StatefulPartitionedCall:P L
'
_output_shapes
:���������.
!
_user_specified_name	input_1:PL
'
_output_shapes
:���������
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
>__forward_critic_layer_call_and_return_conditional_losses_1796
	input_1_0
	input_2_0
	l1_182158
	l1_182160
	l2_182186
	l2_182188
	l3_182213
	l3_182215
identity
l3_statefulpartitionedcall 
l3_statefulpartitionedcall_0
l2_statefulpartitionedcall 
l2_statefulpartitionedcall_0 
l2_statefulpartitionedcall_1
l1_statefulpartitionedcall 
l1_statefulpartitionedcall_0 
l1_statefulpartitionedcall_1
concat_axis
input_1
input_2��L1/StatefulPartitionedCall�L2/StatefulPartitionedCall�L3/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2	input_1_0	input_2_0concat/axis:output:0*
N*
T0*'
_output_shapes
:���������02
concat�
L1/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0	l1_182158	l1_182160*
Tin
2*
Tout
2*W
_output_shapesE
C:���������@:���������@:0@:���������0*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*C
f>R<
:__forward_L1_layer_call_and_return_conditional_losses_17652
L1/StatefulPartitionedCall�
L1/IdentityIdentity#L1/StatefulPartitionedCall:output:0^L1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2
L1/Identity�
L2/StatefulPartitionedCallStatefulPartitionedCallL1/Identity:output:0	l2_182186	l2_182188*
Tin
2*
Tout
2*W
_output_shapesE
C:��������� :��������� :@ :���������@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*C
f>R<
:__forward_L2_layer_call_and_return_conditional_losses_17402
L2/StatefulPartitionedCall�
L2/IdentityIdentity#L2/StatefulPartitionedCall:output:0^L2/StatefulPartitionedCall*
T0*'
_output_shapes
:��������� 2
L2/Identity�
L3/StatefulPartitionedCallStatefulPartitionedCallL2/Identity:output:0	l3_182213	l3_182215*
Tin
2*
Tout
2*D
_output_shapes2
0:���������: :��������� *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*C
f>R<
:__forward_L3_layer_call_and_return_conditional_losses_17172
L3/StatefulPartitionedCall�
L3/IdentityIdentity#L3/StatefulPartitionedCall:output:0^L3/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2
L3/Identity�
IdentityIdentityL3/Identity:output:0^L1/StatefulPartitionedCall^L2/StatefulPartitionedCall^L3/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"#
concat_axisconcat/axis:output:0"
identityIdentity:output:0"
input_1	input_1_0"
input_2	input_2_0"A
l1_statefulpartitionedcall#L1/StatefulPartitionedCall:output:1"C
l1_statefulpartitionedcall_0#L1/StatefulPartitionedCall:output:2"C
l1_statefulpartitionedcall_1#L1/StatefulPartitionedCall:output:3"A
l2_statefulpartitionedcall#L2/StatefulPartitionedCall:output:1"C
l2_statefulpartitionedcall_0#L2/StatefulPartitionedCall:output:2"C
l2_statefulpartitionedcall_1#L2/StatefulPartitionedCall:output:3"A
l3_statefulpartitionedcall#L3/StatefulPartitionedCall:output:1"C
l3_statefulpartitionedcall_0#L3/StatefulPartitionedCall:output:2*Q
_input_shapes@
>:���������.:���������::::::*l
backward_function_nameRP__inference___backward_critic_layer_call_and_return_conditional_losses_1701_179728
L1/StatefulPartitionedCallL1/StatefulPartitionedCall28
L2/StatefulPartitionedCallL2/StatefulPartitionedCall28
L3/StatefulPartitionedCallL3/StatefulPartitionedCall:P L
'
_output_shapes
:���������.
!
_user_specified_name	input_1:PL
'
_output_shapes
:���������
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
;__inference_L1_layer_call_and_return_conditional_losses_803

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:0@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������0:::O K
'
_output_shapes
:���������0
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
?__inference_critic_layer_call_and_return_conditional_losses_870
input_1
input_2
	l1_182158
	l1_182160
	l2_182186
	l2_182188
	l3_182213
	l3_182215
identity��L1/StatefulPartitionedCall�L2/StatefulPartitionedCall�L3/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2input_1input_2concat/axis:output:0*
N*
T0*'
_output_shapes
:���������02
concat�
L1/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0	l1_182158	l1_182160*
Tin
2*
Tout
2*'
_output_shapes
:���������@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*D
f?R=
;__inference_L1_layer_call_and_return_conditional_losses_8032
L1/StatefulPartitionedCall�
L1/IdentityIdentity#L1/StatefulPartitionedCall:output:0^L1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2
L1/Identity�
L2/StatefulPartitionedCallStatefulPartitionedCallL1/Identity:output:0	l2_182186	l2_182188*
Tin
2*
Tout
2*'
_output_shapes
:��������� *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*D
f?R=
;__inference_L2_layer_call_and_return_conditional_losses_8512
L2/StatefulPartitionedCall�
L2/IdentityIdentity#L2/StatefulPartitionedCall:output:0^L2/StatefulPartitionedCall*
T0*'
_output_shapes
:��������� 2
L2/Identity�
L3/StatefulPartitionedCallStatefulPartitionedCallL2/Identity:output:0	l3_182213	l3_182215*
Tin
2*
Tout
2*'
_output_shapes
:���������*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*D
f?R=
;__inference_L3_layer_call_and_return_conditional_losses_7262
L3/StatefulPartitionedCall�
L3/IdentityIdentity#L3/StatefulPartitionedCall:output:0^L3/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2
L3/Identity�
IdentityIdentityL3/Identity:output:0^L1/StatefulPartitionedCall^L2/StatefulPartitionedCall^L3/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:���������.:���������::::::28
L1/StatefulPartitionedCallL1/StatefulPartitionedCall28
L2/StatefulPartitionedCallL2/StatefulPartitionedCall28
L3/StatefulPartitionedCallL3/StatefulPartitionedCall:P L
'
_output_shapes
:���������.
!
_user_specified_name	input_1:PL
'
_output_shapes
:���������
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
;__inference_L3_layer_call_and_return_conditional_losses_726

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� :::O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�

�
:__forward_L1_layer_call_and_return_conditional_losses_1765
inputs_0"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
relu
matmul_readvariableop

inputs��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:0@*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0"
inputsinputs_0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0"
reluRelu:activations:0*.
_input_shapes
:���������0::*h
backward_function_nameNL__inference___backward_L1_layer_call_and_return_conditional_losses_1751_1766:O K
'
_output_shapes
:���������0
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�	
�
'__inference_restored_function_body_1575
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*'
_output_shapes
:���������*(
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_critic_layer_call_and_return_conditional_losses_8702
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:���������.:���������::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������.
!
_user_specified_name	input_1:PL
'
_output_shapes
:���������
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�

�
:__forward_L2_layer_call_and_return_conditional_losses_1740
inputs_0"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
relu
matmul_readvariableop

inputs��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0"
inputsinputs_0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0"
reluRelu:activations:0*.
_input_shapes
:���������@::*h
backward_function_nameNL__inference___backward_L2_layer_call_and_return_conditional_losses_1726_1741:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�"
�
__inference__traced_save_4087
file_prefix/
+savev2_critic_l1_kernel_read_readvariableop-
)savev2_critic_l1_bias_read_readvariableop/
+savev2_critic_l2_kernel_read_readvariableop-
)savev2_critic_l2_bias_read_readvariableop/
+savev2_critic_l3_kernel_read_readvariableop-
)savev2_critic_l3_bias_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const�
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_f331e451ee7a4dfead85991667b4b89b/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
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
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B$l1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"l1/bias/.ATTRIBUTES/VARIABLE_VALUEB$l2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"l2/bias/.ATTRIBUTES/VARIABLE_VALUEB$l3/kernel/.ATTRIBUTES/VARIABLE_VALUEB"l3/bias/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_critic_l1_kernel_read_readvariableop)savev2_critic_l1_bias_read_readvariableop+savev2_critic_l2_kernel_read_readvariableop)savev2_critic_l2_bias_read_readvariableop+savev2_critic_l3_kernel_read_readvariableop)savev2_critic_l3_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes

22
SaveV2�
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard�
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1�
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names�
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity�

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*G
_input_shapes6
4: :0@:@:@ : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:0@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: 
�	
�
"__inference_signature_wrapper_4041
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*'
_output_shapes
:���������*(
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8*(
f#R!
__inference__wrapped_model_40212
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:���������.:���������::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������.
!
_user_specified_name	input_1:PL
'
_output_shapes
:���������
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�	
�
$__inference_critic_layer_call_fn_882
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*'
_output_shapes
:���������*(
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_critic_layer_call_and_return_conditional_losses_8702
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:���������.:���������::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������.
!
_user_specified_name	input_1:PL
'
_output_shapes
:���������
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: "�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_10
serving_default_input_1:0���������.
;
input_20
serving_default_input_2:0���������<
output_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:�E
�
l1
l2
l3

signatures
trainable_variables
	variables
regularization_losses
	keras_api
*/&call_and_return_all_conditional_losses
0_default_save_signature
1__call__"�
_tf_keras_model�{"class_name": "Critic", "name": "critic", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Critic"}}
�

	kernel

bias
trainable_variables
	variables
regularization_losses
	keras_api
*2&call_and_return_all_conditional_losses
3__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "L1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "L1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 48}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 48]}}
�

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*4&call_and_return_all_conditional_losses
5__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "L2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "L2", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 64]}}
�

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*6&call_and_return_all_conditional_losses
7__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "L3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "L3", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 32]}}
,
8serving_default"
signature_map
J
	0

1
2
3
4
5"
trackable_list_wrapper
J
	0

1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
trainable_variables
	variables
layer_regularization_losses
layer_metrics

layers
metrics
non_trainable_variables
regularization_losses
1__call__
0_default_save_signature
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
": 0@2critic/L1/kernel
:@2critic/L1/bias
.
	0

1"
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
trainable_variables
	variables
 layer_regularization_losses
!layer_metrics

"layers
#metrics
$non_trainable_variables
regularization_losses
3__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
": @ 2critic/L2/kernel
: 2critic/L2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
trainable_variables
	variables
%layer_regularization_losses
&layer_metrics

'layers
(metrics
)non_trainable_variables
regularization_losses
5__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
":  2critic/L3/kernel
:2critic/L3/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
trainable_variables
	variables
*layer_regularization_losses
+layer_metrics

,layers
-metrics
.non_trainable_variables
regularization_losses
7__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�2�
?__inference_critic_layer_call_and_return_conditional_losses_870�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *N�K
I�F
!�
input_1���������.
!�
input_2���������
�2�
__inference__wrapped_model_4021�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *N�K
I�F
!�
input_1���������.
!�
input_2���������
�2�
$__inference_critic_layer_call_fn_882�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *N�K
I�F
!�
input_1���������.
!�
input_2���������
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
8B6
"__inference_signature_wrapper_4041input_1input_2�
__inference__wrapped_model_4021�	
X�U
N�K
I�F
!�
input_1���������.
!�
input_2���������
� "3�0
.
output_1"�
output_1����������
?__inference_critic_layer_call_and_return_conditional_losses_870�	
X�U
N�K
I�F
!�
input_1���������.
!�
input_2���������
� "%�"
�
0���������
� �
$__inference_critic_layer_call_fn_882|	
X�U
N�K
I�F
!�
input_1���������.
!�
input_2���������
� "�����������
"__inference_signature_wrapper_4041�	
i�f
� 
_�\
,
input_1!�
input_1���������.
,
input_2!�
input_2���������"3�0
.
output_1"�
output_1���������