# 🤖 **astribot_msgs** | 控制与仿真共用数据结构

**astribot_msgs** 仓库为 Astribot 机器人控制与仿真提供统一的数据结构定义。
**本仓库以子模块形式集成于 [astribot_simulation](https://github.com/Astribot-Dev/astribot_simulation) 与 [astribot_sdk](https://github.com/Astribot-Dev/astribot_sdk/tree/ros1) 项目中，基于 ROS1 Noetic 实现消息通信。

---

## 📦 功能简介

- 提供机器人控制与仿真所需的 ROS 消息定义
- 保证仿真与实际控制的数据结构一致性
- 支持多项目复用，便于维护与扩展

---

## 🖥️ 系统要求

- **操作系统**：Ubuntu 20.04 LTS
- **ROS 版本**：ROS Noetic 

---

## 🚀 快速开始

### 1. 环境准备
- source <your_path>/noetic/setup.bash
- cd <your_path>/astribot_msgs
- catkin_make install

### 2.加载环境
- source <your_path>/astribot_msgs/install/setup.bash
