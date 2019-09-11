from neuralNetWorkClass import neuralNetwork
import matplotlib.pyplot as plt
import numpy
import scipy.ndimage

def training(nw):    
    #打开训练文件
    data_file = open(r"learningData/mnist_train.csv",'r')
    data_list = data_file.readlines()
    data_file.close()

    #画出字符
    #all_value = data_list[95].split(',')
    #image_array = numpy.asfarray(all_value[1:]).reshape((28,28))
    #plt.imshow(image_array,cmap='Greys',interpolation='none')
    #plt.show()
    #print(data_list)
    for all_value in data_list:
        rawData = all_value.split(',')
        #将输入值进行归一化
        scaled_input = (numpy.asfarray(rawData[1:])/255.0*0.99)+0.01
        #正向旋转10°
        #inputsPlusImg = scipy.ndimage.interpolation.rotate(scaled_input.reshape(28,28),10,cval = 0.01,reshape=False)
        #逆向旋转10°
        #inputsMinus10Img = scipy.ndimage.interpolation.rotate(scaled_input.reshape(28,28),-10,cval = 0.01,reshape=False)
        #创建输出目标值
        targets = numpy.zeros(nw.onodes)+0.01
        targets[int(rawData[0])] = 0.99
        #nw.learningNetwork(inputsMinus10Img.reshape(784),targets)
        nw.learningNetwork(scaled_input,targets)


def test_neuralNetwork(nw):
    #测试神经网络权重
    test_data_file = open(r'learningData/mnist_test.csv','r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    #all_value = test_data_list[0].split(',')
    #image_array = numpy.asfarray(all_value[1:]).reshape((28,28))
    #plt.imshow(image_array,cmap='Greys',interpolation='none')
    #print(nw.quaryNetwork(numpy.asfarray(all_value[1:])/255.0*0.99+0.01))
    #plt.show()
    scorecard = []
    for record in test_data_list:
        all_values = record.split(',')
        correct_label = int(all_values[0])
        inputs = (numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
        outputs = nw.quaryNetwork(inputs)
        label = numpy.argmax(outputs)
        if(label == correct_label):
            scorecard.append(1)
            #print(label,"'network's answer")
        else:
            scorecard.append(0)
            #print(label,"'network's answer","But correct answer is",correct_label)
    countPercent = scorecard.count(1)/len(scorecard)*100
    print("final percent is %.2f%%"%countPercent)

def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()







if __name__ == '__main__':
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.3
    n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
    #convert("learningData/train-images.idx3-ubyte", "learningData/train-labels.idx1-ubyte",
    #    "learningData/mnist_train.csv", 60000)
    #convert("learningData/t10k-images.idx3-ubyte", "learningData/t10k-labels.idx1-ubyte",
    #    "learningData/mnist_test.csv", 10000)
    training(n)
    test_neuralNetwork(n)





