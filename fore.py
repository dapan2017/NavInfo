# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 15:45:12 2018
最终版本的长城出发时间预测
得到的文本或者dataframe格式属性为start_longitude,start_latitude,start_time,end_longitude,end_latitude,end_time,start_label,end_label
起点经、纬度，起点时间，终点经、纬度，终点时间，起点标签，终点标签
通过相同的起点经纬度和终点经纬度，找到家到公司和公司到家的行程的时间戳，将这些时间戳转换为每天的时间，进行聚类，
对聚类结果初步分析，界定一个小时之内的为合理预测时间，若是超过一个小时的时间间隔，那么二次聚类，再找出一个小时之内的时间间隔
为用户的出行时间预测，若是没有给出时间段，那说明用用户的出行时间不具有规律性，给出无法预测的结论
@author: fengxue
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from sklearn.cluster import KMeans
from datetime import datetime
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.ensemble import IsolationForest
from collections import Counter

class forecast_Time:
    def __init__(self,IDdata):
        self.Greatwall_data=IDdata
        #挑选家到公司和公司到家的行程出发时间，并将家、公司经纬度和出发时间保存到data字典中
        Home_Company_timestamp,Company_Home_timestamp,Home_Company_time,Company_Home_time=[],[],[],[]
        Home=(0,0)
        Company=(0,0)
        for i in range(len(self.Greatwall_data)):
            if self.Greatwall_data.start_label[i]==1 and self.Greatwall_data.end_label[i]==2:
                Home=(self.Greatwall_data.start_longitude[i],self.Greatwall_data.start_latitude[i])
                Company=(self.Greatwall_data.end_longitude[i],self.Greatwall_data.end_latitude[i])
                Home_Company_timestamp.append(self.Greatwall_data.start_time[i])
                Home_Company_time.append(time.localtime(self.Greatwall_data.start_time[i]).tm_hour*60+time.localtime(self.Greatwall_data.start_time[i]).tm_min)
            elif self.Greatwall_data.start_label[i]==2 and self.Greatwall_data.end_label[i]==1:
                Home=(self.Greatwall_data.start_longitude[i],self.Greatwall_data.start_latitude[i])
                Company=(self.Greatwall_data.end_longitude[i],self.Greatwall_data.end_latitude[i])
                Company_Home_timestamp.append(self.Greatwall_data.start_time[i])
                Company_Home_time.append(time.localtime(self.Greatwall_data.start_time[i]).tm_hour*60+time.localtime(self.Greatwall_data.start_time[i]).tm_min)
        self.data={'Home':Home,'Company':Company,'Home_Company_timestamp':Home_Company_timestamp,'Home_Company_time':Home_Company_time,'Company_Home_timestamp':Company_Home_timestamp,'Company_Home_time':Company_Home_time}
        

    def foreTime(self):
    #归一化函数，将一天的时间全部以一天1440分钟为尺寸来归一化
        def Standard(l_Original):
            l_stand=[[l_Original[i]/1440] for i in range(len(l_Original))]
            return l_stand
    #聚类函数，返回聚类的结果、聚类后每个类的信息（时间段、频次、概率）、时间的方差，预测时间段            
        def cluster(timestamp,depart_time):
            l=[[depart_time[i]] for i in range(len(depart_time))]
            l_stand=Standard(depart_time)
            var_time=np.var(l_stand)
    #密度聚类
            db = DBSCAN(eps=0.11, min_samples=3)
            db.fit(l_stand)
            label=db.labels_.tolist()
    #对聚类结果分标签分类，对每个类进行统计分析，统计出每个类的频次和概率，将概率最大的那个时间段输出作为用户的出行时间
            cluster_data=pd.DataFrame([timestamp,depart_time,label],index=['Timestamp','Start_time','Label']).T
            result,probability=[],[]
            for name,group in cluster_data.groupby(['Label']):
                group=group.sort_values(by=['Start_time'])
                result.append([group,time.strftime("%Y--%m--%d %H:%M:%S",time.localtime(group.Timestamp.iloc[0]))[-8:-3],group.Start_time.iloc[0],time.strftime("%Y--%m--%d %H:%M:%S", time.localtime(group.Timestamp.iloc[len(group)-1]))[-8:-3],group.Start_time.iloc[len(group)-1],len(group),len(group)/len(cluster_data)])       
                probability.append(len(group)/len(cluster_data))
            probability_index=probability.index(max(probability))
            out_result=result[probability_index]
            #对初次聚类的结果进行再次分析，若是初次聚类的出发时间间隔是一个小时之内，则认为合格，若是大于60min小于350min再进行二次聚类，将二次聚类后概率大的那部分作为结果输出，对于大于350min的用户则认为出行不规律，无法做出出发时间的预测
            out_result_time=list(out_result[0].Start_time)
            time_interval=max(out_result_time)-min(out_result_time)
            if time_interval<60:
                end_out_result=[out_result[1],out_result[3],out_result[6]]
            elif 60<time_interval<250:
                out_result_time=[[out_result_time[i]] for i in range(len(out_result_time))]
                n_cluster=2
                estimator=KMeans(n_clusters=n_cluster)   
                estimator.fit(out_result_time)
                out_result_time_label=estimator.labels_.tolist()
                most_Label=Counter(out_result_time_label).most_common(1)[0][0]
                most_Label_index=[i for i,v in enumerate(out_result_time_label) if v==most_Label]
                l=[out_result_time[i][0] for i in most_Label_index]
                end_out_result=[str(int(min(l)/60))+':'+'%02d'%(int(min(l)%60)),str(int(max(l)/60))+':'+'%02d'%(int(max(l)%60)),len(l)/len(cluster_data)]                
            else:
                end_out_result=['it can not predict'] 
            #结果进行输出，输出字段为：home到company的时间段，company到home的时间段，HC的方差，CH的方差，HC预测出发时间，CH预测出发时间
            return cluster_data,result,var_time,end_out_result
        #对结果分类处理，对于没有家和公司经纬度，缺一个行程的分别进行时间预测，两个行程都有的预测
        if self.data['Home']==(0,0):
            HC_Time=['it can not Predict']
            CH_Time=['it can not Predict']
        elif self.data['Home_Company_timestamp']==[]:
            HC_Time=['it can not predict']
            Company_Home_time,CH_result,CH_var_time,CH_Time=cluster(self.data['Company_Home_timestamp'],self.data['Company_Home_time'])
            self.data.update({'CH_result':CH_result,'CH_var_time':CH_var_time,'CH_Time':CH_Time})
        elif self.data['Company_Home_timestamp']==[]:
             CH_Time=['it can not predict']
             Home_Company_time,HC_result,HC_var_time,HC_Time=cluster(self.data['Home_Company_timestamp'],self.data['Home_Company_time'])
             self.data.update({'HC_result':HC_result,'HC_var_time':HC_var_time,'HC_Time':HC_Time})
        else:
            Home_Company_time,HC_result,HC_var_time,HC_Time=cluster(self.data['Home_Company_timestamp'],self.data['Home_Company_time'])
            Company_Home_time,CH_result,CH_var_time,CH_Time=cluster(self.data['Company_Home_timestamp'],self.data['Company_Home_time'])
            self.data.update({'HC_result':HC_result,'HC_var_time':HC_var_time,'HC_Time':HC_Time,'CH_result':CH_result,'CH_var_time':CH_var_time,'CH_Time':CH_Time})
#        return HC_Time,CH_Time,self.data
        #返回最终的结果，数据包括家的经纬度，公司的经纬度，家到公司的出发时间段和置信度，公司到家的出发时间和置信度
        return {'Home_Company_time':self.data['Home_Company_time'],'Company_Home_time':self.data['Company_Home_time'],'Home_loca':self.data['Home'],'Company_loca':self.data['Company'],'HomeToCompany_Time':HC_Time,'CompanyToHome_Time':CH_Time}
#        return {'Home_loca':self.data['Home'],'Company_loca':self.data['Company'],'HomeToCompany_Time':HC_Time,'CompanyToHome_Time':CH_Time}


