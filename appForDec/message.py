
import sys
from PyQt5 import QtWidgets

class MessageBox(QtWidgets.QWidget):#�̳��Ը���QtWidgets.QWidget
    def __init__(self,parent = None):#parent = None�����QWidget�������ϲ�Ĵ���,Ҳ����MainWindows.
        QtWidgets.QWidget.__init__(self)#��Ϊ�̳й�ϵ��Ҫ�Ը����ʼ��
#ͨ��super��ʼ�����࣬__init__()������self����ֱ��QtWidgets.QWidget.__init__(self)������������self��
        self.setGeometry(300, 300, 1000,1000)  # setGeometry()���������������--���ô�������Ļ�ϵ�λ�ú����ô��ڱ���Ĵ�С������ǰ���������Ǵ�������Ļ�ϵ�x��y���ꡣ�����������Ǵ��ڱ���Ŀ�͸�
        self.setWindowTitle(u'����')  # ���ô�����⣬���п��п��ޡ�
        self.button = QtWidgets.QPushButton(u'����', self)  # ����һ����ť��ʾ�����ԡ�����
        self.button.move(300,300)
        self.button.clicked.connect(self.show_message)  # �źŲ�

    def show_message(self):



