
# coding: utf-8

# In[1]:


from build_dense import *
from functions import *
import matplotlib.pyplot as plt

x = np.linspace(0,2,51200)
y_saw = saw_wave(0.3,1,x)
y_square = square_wave(0.4,1,x)
y_sine = sine_wave(0.4,1,x)
y_sine_saw = y_sine + y_saw

y_sine0 = sine_wave(0.2,1,x)
y_sine1 = sine_wave(0.3,1,x)
y_sine2 = sine_wave(0.5,1,x)
y_sine3 = sine_wave(0.7,1,x)
y_sine4 = sine_wave(1.1,1,x)
y_sine5 = sine_wave(1.3,1,x)
y_sine6 = sine_wave(1.7,1,x)
y_sine7 = sine_wave(1.9,1,x)
y_multi_sine = y_sine0 + y_sine1 + y_sine2 + y_sine3 + y_sine4 + y_sine5 + y_sine6 + y_sine7

y_list = [y_saw, y_square, y_sine, y_sine_saw, y_multi_sine]
name_list = ['saw_wave', 'square_wave', 'sine', 'sine_saw', 'multi_sine_2']


# In[2]:


for i in range(len(y_list)):
    plt.plot(x,(y_list[i]))
    plt.title(name_list[i])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('report_data/function/' + name_list[i] + '/function.png', dpi = 300)
    plt.show()


# In[3]:


def build_model_list():
    model_1 = build_dense(1, [335])
    model_4 = build_dense(4, [12, 18, 25, 10])
    model_10 = build_dense(10, [5, 7, 9, 11, 13, 13, 11, 9, 9, 7])
    return [model_1, model_4, model_10]


# In[7]:


for i in range(1):
    print('function', name_list[i], i)
    y = y_list[i]
    model_list = build_model_list()
    hist_list = [0,0,0]
    for j in range(3):
        print('training model', j)
        hist_list[j] = train(model_list[j], x, y, 1024, 5000, 0)
    #loss
    plot1 = plt.plot(hist_list[0].history['loss'], 'b-', label = 'model_1')
    plot2 = plt.plot(hist_list[1].history['loss'], 'g-', label = 'model_4')
    plot3 = plt.plot(hist_list[2].history['loss'], 'r-', label = 'model_10')
    plt.title('error')
    plt.xlabel('epochs')
    plt.ylabel('error')
    plt.legend()
    plt.savefig('report_data/function/' + name_list[i] + '/loss.png', dpi = 300)
    plt.show()
    #loss log10
    plot1 = plt.plot(np.log10(hist_list[0].history['loss']), 'b-', label = 'model_1')
    plot2 = plt.plot(np.log10(hist_list[1].history['loss']), 'g-', label = 'model_4')
    plot3 = plt.plot(np.log10(hist_list[2].history['loss']), 'r-', label = 'model_10')
    plt.title('error with log10')
    plt.xlabel('epochs')
    plt.ylabel('error')
    plt.legend()
    plt.savefig('report_data/function/' + name_list[i] + '/loss_log10.png', dpi = 300)
    plt.show()
    #predict
    predict_list = [0,0,0]
    for j in range(3):
        predict_list[j] = model_list[j].predict(x)
    plt.plot(y, 'c-', label = 'function')
    plot1 = plt.plot(predict_list[0], 'b-', label = 'model_1')
    plot2 = plt.plot(predict_list[1], 'g-', label = 'model_4')
    plot3 = plt.plot(predict_list[2], 'r-', label = 'model_10')
    plt.title('predict')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('report_data/function/' + name_list[i] + '/simulate.png', dpi = 300)
    plt.show()


# In[8]:


for i in model_list:
    i.summary()
    print()

