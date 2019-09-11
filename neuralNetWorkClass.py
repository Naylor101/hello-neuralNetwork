import numpy
import scipy.special

class neuralNetwork:
    #初始化神经网络
    def __init__(self,inputNodes,hiddenNodes,outputNodes,learningRate):
        #定义输入，隐藏输出层节点数量
        self.inodes = inputNodes
        self.onodes = outputNodes
        self.hnodes = hiddenNodes
        self.lrt = learningRate
        print("init success")
        #定义权重矩阵
        self.wih = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        
        self.who = numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        #定义平滑函数
        self.activation_function = lambda x:scipy.special.expit(x)


        pass
    #训练神经网络
    def learningNetwork(self,input_lists,target_list):
        #转化输入矩阵为转置矩阵
        inputs = numpy.array(input_lists,ndmin=2).T
        #转化目标矩阵为转置矩阵
        targets = numpy.array(target_list,ndmin=2).T

        #计算到隐藏层的信号
        hidden_inputs = numpy.dot(self.wih,inputs)
        #计算隐藏层的输出信号
        hidden_outputs = self.activation_function(hidden_inputs)
        #计算信号到最终输出层
        final_inputs = numpy.dot(self.who,hidden_outputs)
        #计算最终输出层的激活信号
        final_outputs = self.activation_function(final_inputs)



        #计算输出层的误差
        output_errors = targets - final_outputs
        #计算隐藏层的误差
        hidden_errors = numpy.dot(self.who.T,output_errors)
        #更新隐藏层和输出层之间的权重
        self.who += self.lrt * numpy.dot((output_errors * final_outputs *(1.0-final_outputs)),numpy.transpose(hidden_outputs))
        #更新输入层和隐藏层之间的权重
        self.wih += self.lrt * numpy.dot((hidden_errors * hidden_outputs *(1.0 - hidden_outputs)),numpy.transpose(inputs))



        pass




    #查询神经网络
    def quaryNetwork(self,inputs_list):
        #转化输入矩阵为转置矩阵
        inputs = numpy.array(inputs_list,ndmin=2).T
        #计算到隐藏层的信号
        hidden_inputs = numpy.dot(self.wih,inputs)
        #计算隐藏层的输出信号
        hidden_outputs = self.activation_function(hidden_inputs)
        #计算信号到最终输出层
        final_inputs = numpy.dot(self.who,hidden_outputs)
        #计算最终输出层的激活信号
        final_outputs = self.activation_function(final_inputs)
        #返回最终输出
        return final_outputs
        