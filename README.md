# PlateletPro
生物医药数据处理集成工具使用说明书 
目录 
1. 软件概述 
2. 安装与启动 
3. 界面介绍 
4. 基本操作流程 
5. 支持的文件类型 
6. 处理模块详解 
TEG 血栓弹力图数据 
血小板聚集仪(AA)数据 
Transwell 细胞穿膜实验 
视频处理功能 
分子结构文件转换 
酶标仪数据处理 
GROMACS XVG文件转换 
7. 输出文件说明 
8. 常见问题与解决方案 
9. 联系与支持 
1. 软件概述 
本软件是为生物医药实验研究领域设计的数据处理集成工具，旨在简化实验数据的处理流程，提高数据处理效率和标准化程度。软件支持多种实验类型的数据处理，包括血栓弹力图(TEG)、血小板聚集分析、细胞穿膜实验、视频处理等。 
该工具采用图形化界面设计，无需编程知识即可操作，适合实验室科研人员和技术人员使用。处理后的数据以标准化格式输出，便于后续的统计分析和图表制作。 
2. 安装与启动 
系统要求 
Windows 7/8/10/11 操作系统 
至少 4GB RAM 
至少 500MB 可用磁盘空间 
安装步骤 
下载软件安装包(.exe文件) 
双击安装包文件运行安装程序 
按照安装向导的提示完成安装 
启动软件 
从桌面快捷方式或开始菜单中启动软件 
双击应用程序图标即可启动主界面 
3. 界面介绍 
软件启动后显示的主界面包含以下关键元素： 
文件夹设置区域：包含输入和输出文件夹的选择按钮和路径显示 
处理按钮：开始执行数据处理的主按钮 
退出按钮：关闭程序 
日志区域：显示处理过程中的状态信息和错误提示 
进度条：显示当前处理进度 
4. 基本操作流程 
选择输入文件夹：点击"选择输入文件夹"按钮，选择包含待处理数据文件的文件夹 
选择输出文件夹：点击"选择输出文件夹"按钮，选择处理后文件的保存位置 
开始处理：点击"开始处理"按钮启动自动处理流程 
查看处理进度：通过进度条和日志信息跟踪处理状态 
查看结果：处理完成后，在指定的输出文件夹中查看生成的文件 
查看处理报告：处理完成后，会弹出处理结果统计窗口 
5. 支持的文件类型 
本软件能够自动识别并处理以下类型的文件： 
文件扩展名
数据类型
处理方式
.txt
TEG血栓弹力图数据
转换为CSV格式并进行可视化处理
.xlsx, .xls, .xlsm
血小板聚集仪(AA)数据或酶标仪数据
根据数据特征进行识别和相应处理
.tif, .jpg, .jpeg
Transwell细胞穿膜实验图像
分析紫色区域占比并生成结果文件
.avi
视频文件
转换为MP4格式
.mov, .mp4
视频文件
提取关键帧并进行荧光强度分析
.cif
分子结构文件
转换为PDB格式
.xvg
GROMACS分子动力学模拟输出
转换为CSV格式
.csv
多种类型数据
根据内容特征进行识别和处理

6. 处理模块详解 
TEG 血栓弹力图数据 
支持文件类型：.txt 
处理流程： 
读取原始TEG数据文件 
提取x和y坐标数据 
添加z轴数据(y轴数据的负值) 
转换为标准CSV格式 
在可视化步骤中合并同一文件夹内的多个TEG文件 
输出文件： 
单个文件处理结果：output-[原文件名]_teg.csv 
文件夹汇总结果：visualized-[文件夹名]_teg.csv 
血小板聚集仪(AA)数据 
支持文件类型：.xlsx, .xls, .xlsm, .csv 
识别特征： 
第一行值为"NjData" 
第二行值为"ADPrateData" 
处理流程： 
识别特定格式的血小板聚集仪数据 
移除前两行标题信息 
处理特殊分隔符"@#" 
转置数据结构 
输出为标准Excel格式 
输出文件：output-[原文件名].xlsx 
Transwell 细胞穿膜实验 
支持文件类型：.tif, .jpg, .jpeg 
处理流程： 
读取图像文件 
转换为HSV颜色空间 
检测并计算紫色区域占比 
生成包含文件名和紫色占比的CSV文件 
汇总同一文件夹内的所有Transwell结果 
输出文件： 
单个文件处理结果：output-[原文件名]_transwell.csv 
文件夹汇总结果：visualized-[文件夹名]_transwell.csv 
全局汇总结果：summarized-transwell.csv 
视频处理功能 
视频格式转换： 
支持文件类型：.avi 
处理方式：转换为MP4格式 
输出文件：output-[原文件名]_a2m.mp4 
荧光强度分析(FA)： 
支持文件类型：.mov, .mp4 (小于10分钟的视频) 
处理流程： 
自动检测视频时长 
对视频中五个固定区域(左上、右上、左下、右下、中心)进行荧光强度分析 
每秒采样一次，计算平均强度 
生成时间序列数据 
输出文件：output-[原文件名]_fa.csv 
视频截图提取： 
支持文件类型：.mov, .mp4 
处理流程： 
根据视频时长确定时间点： 
小于5分钟：1, 2, 3分钟 
5-10分钟：1, 3, 5分钟 
大于10分钟：3, 6, 9, 12, 15分钟 
在指定时间点提取视频帧 
保存为JPEG图像 
输出文件：保存在以原视频名命名的子文件夹中，格式为[视频名]_[分钟]min.jpg 
分子结构文件转换 
支持文件类型：.cif 
处理流程： 
使用Open Babel工具将CIF格式转换为PDB格式 
保留原始文件的分子结构信息 
输出文件：output-[原文件名].pdb 
酶标仪数据处理 
支持文件类型：.xlsx, .xls, .xlsm, .csv 
处理流程： 
读取96孔板格式的原始数据 
重组数据结构为标准表格形式 
适当处理行列结构，便于后续分析 
输出文件：output-[原文件名].xlsx 
GROMACS XVG文件转换 
支持文件类型：.xvg 
处理流程： 
读取XVG文件内容 
过滤掉以@或#开头的注释行 
提取数值数据 
转换为CSV格式 
输出文件：保存在"xvg_csv"子文件夹中，格式为[当前文件夹名]_[原文件名].csv 
7. 输出文件说明 
处理后的文件会按照以下规则命名并保存： 
单个文件处理结果：前缀为"output-"，后跟原文件名和处理类型标识 
例如：output-sample_teg.csv、output-experiment_fa.csv 
可视化汇总文件：前缀为"visualized-"，后跟文件夹名和处理类型标识 
例如：visualized-experiment1_teg.csv、visualized-test2_transwell.csv 
全局汇总文件：前缀为"summarized-"，后跟处理类型标识 
例如：summarized-transwell.csv 
特殊处理文件： 
视频截图：保存在以原视频名命名的子文件夹中 
XVG转换结果：保存在"xvg_csv"子文件夹中 
8. 常见问题与解决方案 
Q: 软件无法识别我的文件类型 
A: 请确认文件扩展名正确，并且文件内容符合相应格式要求。对于特殊格式的文件，可能需要预先进行格式转换。 
Q: 处理过程中出现错误 
A: 查看日志区域的错误信息，常见原因包括：文件格式不正确、文件损坏、或缺少必要的数据列。尝试检查原始文件并修复后重新处理。 
Q: 输出文件中数据不完整 
A: 可能是原始数据中包含异常值或格式不一致。检查原始文件，确保数据格式符合处理要求。 
Q: 视频处理速度较慢 
A: 视频处理通常较为耗时，特别是高分辨率或长时间的视频。请耐心等待处理完成。 
Q: Open Babel相关错误 
A: 分子结构文件转换依赖于Open Babel工具。请确保系统中正确安装了Open Babel，并已添加到系统PATH中。 
9. 联系与支持 
如遇到无法解决的问题，或有功能改进建议，请联系： 
作者：王天宇，周绍芸，申传斌 
邮箱：Terrywangtianyu@gmail.com 
单位：中国海洋大学血液与心脑血管药理课题组 (OUC Blood and Cardiovascular Pharmacology Research Group) 
