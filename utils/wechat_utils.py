#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Time : 2019/1/9 16:07
import json
import time

import requests
import urllib3

__author__ = 'hua.guo'

def SendHeartBeat(agCode):
    heartBeatUrl = "http://10.1.33.73/monitor/api/beat/"+agCode
    http = urllib3.PoolManager()
    heartBeatResp = http.request('POST', heartBeatUrl)
    # headers = {'Content-Type': 'application/json'}
    # heartBeatResp = requests.post(url=heartBeatUrl, headers=headers)
    if heartBeatResp == 200:
        logger.info('send heart beat finished!')
        return 0
    else:
        return 1
        logger.info('send heart beat failed!')

def SendHeartBeat_GBK(agCode):
    heartBeatUrl = "http://10.1.33.73/monitor/api/beat/"+agCode.decode('gbk').encode('utf-8')
    heartBeatResp =  requests.get(heartBeatUrl)
    if heartBeatResp.status_code == 200:
        return 0
    else:
        return 1

def SendErrorMsg(agCode,weCode,faultLevel,errorCode,eventMsg):
    eventMsgUrl = "http://10.1.33.73/monitor/api/msg"
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    eventData = {"agCode":agCode, "weCode":weCode, "eventTime":now}
    eventData["faultLevel"] = faultLevel
    eventData["errorCode"] = errorCode
    eventData["eventMsg"] = eventMsg
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url=eventMsgUrl, headers=headers, data=json.dumps(eventData).encode("utf-8"))
    if response == 200:
        return 0
    else:
        return 1

def SendAnyMessage(agCode,weCode,weName,faultLevel,errorCode,eventMsg,weReceiver,weSenderType,weSystem,weType,msgLabel,weDesc):
    eventMsgUrl = "http://10.1.33.73/monitor/api/v2/msg"
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    eventData = {"agCode":agCode, "weCode":weCode, "eventTime":now}
    eventData["faultLevel"] = faultLevel
    eventData["errorCode"] = errorCode
    eventData["eventMsg"] = eventMsg
    eventData["weName"] = weName
    eventData["weReceiver"] = weReceiver
    eventData["weSenderType"] = weSenderType
    eventData["weSystem"] = weSystem
    eventData["weType"] = weType
    eventData["msgLabel"] = msgLabel
    eventData["weDesc"] = weDesc
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url=eventMsgUrl, headers=headers, data=json.dumps(eventData).encode("utf-8"))
    if response.status_code == 200:
        return 0
    else:
        return 1

def SendAnyMessage_GBK(agCode,weCode,weName,faultLevel,errorCode,eventMsg,weReceiver,weSenderType,weSystem,weType,msgLabel,weDesc):
    eventMsgUrl = "http://10.1.33.73/monitor/api/v2/msg"
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    eventData = {"agCode":agCode.decode('gbk').encode('utf-8'), "weCode":weCode.decode('gbk').encode('utf-8'), "eventTime":now}
    eventData["faultLevel"] = faultLevel.decode('gbk').encode('utf-8')
    eventData["errorCode"] = errorCode.decode('gbk').encode('utf-8')
    eventData["eventMsg"] = eventMsg.decode('gbk').encode('utf-8')
    eventData["weName"] = weName.decode('gbk').encode('utf-8')
    eventData["weReceiver"] = weReceiver.decode('gbk').encode('utf-8')
    eventData["weSenderType"] = weSenderType.decode('gbk').encode('utf-8')
    eventData["weSystem"] = weSystem.decode('gbk').encode('utf-8')
    eventData["weType"] = weType.decode('gbk').encode('utf-8')
    eventData["msgLabel"] = msgLabel.decode('gbk').encode('utf-8')
    eventData["weDesc"] = weDesc.decode('gbk').encode('utf-8')
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url=eventMsgUrl, headers=headers, data=json.dumps(eventData).encode("utf-8"))
    if response.status_code == 200:
        return 0
    else:
        return 1

def SendJcsMessage(weName, eventMsg, weReceiver, weSenderType, weDesc, msgLabel='业务监控任务'):
    agCode = 'AG_JCS_JOB'
    faultLevel = 'FAULT_ERROR'
    errorCode = 'ERROR'
    weSystem ='JCS'
    weType = 'ET_BUSINESS'
    #最长消息不超过2000个字符
    eventMsg = str(eventMsg)[0:2000]
    weCode = str(weName)
    SendAnyMessage(agCode, weCode, weName, faultLevel, errorCode, eventMsg, str(weReceiver), str(weSenderType), weSystem, weType, msgLabel, str(weDesc))

def SendJcsInfoMessage(weName, eventMsg, weReceiver, weSenderType, weDesc, msgLabel='业务监控任务'):
    agCode = 'AG_JCS_JOB'
    faultLevel = 'FAULT_NORM'
    errorCode = 'INFO'
    weSystem ='JCS'
    weType = 'ET_BUSINESS'
    #最长消息不超过2000个字符
    eventMsg = str(eventMsg)[0:2000]
    weCode = str(weName)
    SendAnyMessage(agCode, weCode, str(weName), faultLevel, errorCode, eventMsg, str(weReceiver), str(weSenderType), weSystem, weType, msgLabel, str(weDesc))

def MyTestFunc():
    return SendAnyMessage("AG_JCS_JOB","JCS_JOB_TEST","测试","FAULT_ERROR","ERROR","测试失败","hua.guo","ST_EMAIL;ST_WECHAT","JCS","ET_BUSINESS","网络设备","测试一下")

if __name__ == "__main__":
    SendJcsMessage('测试1', 'ceshi2', 'hua.guo', 'ST_EMAIL;ST_WECHAT', 'ceshi3')
    # SendHeartBeat('AG_JCS_JOB')