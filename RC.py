import sys
import gc
import os
import pandas as pd
import numpy as np
import csv
import subprocess
import openpyxl
import cv2
import natsort
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QPushButton,
    QLabel,
    QVBoxLayout,
    QWidget,
    QProgressBar,
    QHBoxLayout,
    QTextEdit,
    QGroupBox,
    QMessageBox,
    QDialog,
    QTextBrowser
)
from PyQt6.QtCore import (
    QThread,
    pyqtSignal,
    Qt
)

# release_memory 装饰器函数
# 用于手动触发垃圾回收，释放内存资源
# 在函数执行完成后调用gc.collect()，防止大型文件处理时内存泄漏
# 定义一个装饰器，用于释放内存资源
def release_memory(func):
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            gc.collect()  # 手动触发垃圾回收
            return result
        except Exception as e:
            print(f"Memory release error: {e}")
            return None
    return wrapper


# --- reserved for extension ---
# 预留的插件基础类，可扩展
class PluginBase:
    """预留插件基础类"""
    def __init__(self):
        pass

# 预留的插件基础类，可扩展
class VideoProcessorPlugin(PluginBase):
    def process(self, video_path):
        pass
# --- reserved for extension ---
# FileProcessorThread.run 方法
# 文件处理线程的主执行方法
# 初始化文件计数器和错误列表，然后递归处理所有文件夹
class FileProcessorThread(QThread):
    update_progress = pyqtSignal(int)
    update_log = pyqtSignal(str)
    processing_completed = pyqtSignal(list, dict, list)

    def __init__(self, input_folder, output_folder):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder

    def run(self):
        # 初始化文件类型计数器和错误文件列表
        file_type_counts = {
            'TEG': 0,
            'AA': 0,
            'Transwell': 0,
            'AVI2MP4': 0,
            'MOV_MP4': 0,
            'Excel_CSV': 0,
            'cif2pdb':0,
            'video2pic':0,
            'xvg2csv':0,
            'Total': 0
        }
        error_files = []
        processed_files = []

        # 调用递归处理方法
        self.recursive_process_folder(
            self.input_folder,
            self.output_folder,
            file_type_counts,
            error_files,
            processed_files,
            initial_input_folder=self.input_folder
        )

        # 发送处理结果
        self.processing_completed.emit(processed_files, file_type_counts, error_files)

    # FileProcessorThread.recursive_process_folder 方法
    # 递归处理文件夹中的所有文件和子文件夹
    # 根据文件扩展名调用相应的处理函数，并更新处理进度和日志
    def recursive_process_folder(self, input_folder, output_folder, file_type_counts, error_files, processed_files,
                                 initial_input_folder):
        # 更新日志，显示当前处理的文件夹
        self.update_log.emit(f"\n开始处理文件夹: {input_folder}")

        # 1. 检查是否有次级文件夹
        all_items = os.listdir(input_folder)
        subfolders = [f for f in all_items if os.path.isdir(os.path.join(input_folder, f))]

        # 2.1 如果有次级文件夹，递归处理
        if subfolders:
            for subfolder in subfolders:
                new_input_path = os.path.join(input_folder, subfolder)
                new_output_path = os.path.join(output_folder, subfolder)
                os.makedirs(new_output_path, exist_ok=True)

                # 递归调用子文件夹
                self.recursive_process_folder(
                    new_input_path,
                    new_output_path,
                    file_type_counts,
                    error_files,
                    processed_files,
                    initial_input_folder
                )

        # 2.2 处理当前文件夹中的文件
        # 重置进度条为0%
        self.update_progress.emit(0)

        # 获取当前文件夹中的可处理文件
        processable_files = [
            f for f in os.listdir(input_folder)
            if os.path.isfile(os.path.join(input_folder, f)) and
               f.endswith(('.tif', '.tiff', '.mov', '.mp4', '.avi', '.txt', '.xlsx', '.xls', '.xlsm', '.csv', '.cif', '.xvg')) and
               not f.startswith(('visualized-', 'summarized-'))
        ]

        total_files = len(processable_files)
        if total_files == 0:
            self.update_log.emit(f"文件夹 {input_folder} 中没有可处理的文件")
            return

        # 处理当前文件夹中的每个文件
        for index, file in enumerate(processable_files, 1):
            try:
                file_path = os.path.join(input_folder, file)
                output_file_path = os.path.join(output_folder, f"output-{os.path.splitext(file)[0]}")

                self.update_log.emit(f"正在处理文件: {file} ({index}/{total_files})")

                if file.endswith('.xvg'):
                    # 构造输出文件夹：可以将转换后的文件存放在当前文件夹对应的输出路径下
                    # 例如，对于每个 xvg 文件，都在当前 output_folder 下建立一个子目录 "xvg_csv" 用于保存转换结果
                    xvg_output_dir = os.path.join(output_folder, "xvg_csv")
                    xvg2csv(input_folder, xvg_output_dir)
                    file_type_counts['xvg2csv'] += 1
                    # 对其他文件类型的判断……
                elif file.endswith(('.mov', '.mp4')):
                    # 打开视频文件，获取 fps 和总帧数来计算时长
                    cap = cv2.VideoCapture(file_path)
                    if not cap.isOpened():
                        self.update_log.emit(f"无法打开视频文件: {file}")
                        continue  # 跳过处理

                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    duration = total_frames / fps if fps else 0

                    cap.release()  # 检测完毕后及时释放资源

                    # 判断视频时长是否小于10分钟（600秒）
                    if duration < 600:
                        output_file_path += "_fa.csv"
                        self.update_log.emit(f"处理视频: {file} 时长 {duration:.2f}s 小于10min，进行FA处理")
                        process_fa(file_path, output_file_path)
                        file_type_counts['MOV_MP4'] += 1
                    else:
                        self.update_log.emit(f"视频 {file} 时长 {duration:.2f}s 超过10min，跳过MFI处理")
                    cap = cv2.VideoCapture(file_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    duration = total_frames / fps if fps else 0
                    cap.release()

                    # 根据时长确定时间点（注意这里的时间点单位均为分钟）
                    if duration < 300:  # 小于5分钟
                        timestamps = [1, 2, 3]
                    elif duration < 600:  # 5到10分钟
                        timestamps = [1, 3, 5]
                    else:  # 超过10分钟
                        timestamps = [3, 6, 9, 12, 15]

                    # 调用 process_video_screenshots 来生成截图
                    process_video_screenshots(file_path, output_folder, timestamps)
                    file_type_counts['video2pic'] += 1
                elif file.endswith('.avi'):
                    output_file_path += "_a2m.mp4"
                    process_avi2mp4(file_path, output_file_path)
                    file_type_counts['AVI2MP4'] += 1
                elif file.endswith('.txt'):
                    output_file_path += "_teg.csv"
                    process_teg(file_path, output_file_path)
                    file_type_counts['TEG'] += 1
                elif file.endswith('.cif'):
                    output_file_path = os.path.splitext(os.path.join(output_folder, f"output-{os.path.splitext(file)[0]}"))[0] + ".pdb"
                    process_cif2pdb(file_path, output_file_path)
                    file_type_counts['cif2pdb'] += 1
                elif file.endswith('.xvg'):
                    output_file_path += "_.csv"
                    process_teg(file_path, output_file_path)
                    file_type_counts['xvg2csv'] += 1
                elif file.endswith(('.tif', '.jpg', '.jpeg')):
                    output_file_path += "_transwell.csv"
                    process_transwell(file_path, output_file_path)
                    file_type_counts['Transwell'] += 1
                elif file.endswith(('.xlsx', '.xls', '.xlsm', '.csv')):
                    output_file_path += ".xlsx"

                    if file.endswith('.csv'):
                        df = pd.read_csv(file_path, header=None)
                    else:
                        df = pd.read_excel(file_path, header=None)

                    first_value = df.iloc[0, 0] if not df.empty else None
                    second_value = df.iloc[1, 0] if df.shape[0] > 1 else None

                    if first_value == 'NjData' and second_value == 'ADPrateData':
                        process_aa(file_path, output_file_path)
                        file_type_counts['AA'] += 1
                    else:
                        process_mr(file_path, output_file_path)
                        file_type_counts['Excel_CSV'] += 1

                processed_files.append(output_file_path)
                file_type_counts['Total'] += 1

                # 更新进度条 - 使用当前文件夹的进度
                progress = int((index / total_files) * 100)
                self.update_progress.emit(progress)

            except Exception as e:
                error_info = {'file': file_path, 'error_message': str(e)}
                error_files.append(error_info)
                self.update_log.emit(f"处理文件 {file} 时出错: {str(e)}")

        # 处理可视化和汇总
        if input_folder == initial_input_folder:
            # TEG文件可视化
            teg_files = [f for f in processed_files if f.endswith('teg.csv')]
            if teg_files:
                self.update_log.emit("正在生成TEG可视化文件...")
                teg_visualized_files = visualize_teg_files(input_folder, output_folder)
                processed_files.extend(teg_visualized_files)

            # Transwell文件可视化和汇总
            transwell_files = [f for f in processed_files if f.endswith('transwell.csv')]
            if transwell_files:
                self.update_log.emit("正在生成Transwell可视化文件...")
                transwell_visualized_files = visualize_transwell_files(input_folder, output_folder)
                processed_files.extend(transwell_visualized_files)
                summarize_transwell_files(output_folder)

            # 添加FA文件可视化
            fa_files = [f for f in processed_files if f.endswith('fa.csv')]
            if fa_files:
                self.update_log.emit("正在生成FA可视化文件...")
                fa_visualized_files = visualize_fa_files(input_folder, output_folder)
                processed_files.extend(fa_visualized_files)

        self.update_log.emit(f"文件夹 {input_folder} 处理完成")
        # 确保当前文件夹处理完成时进度条显示100%
        self.update_progress.emit(100)




class FileProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    # FileProcessorApp.initUI 方法
    # 初始化应用程序的用户界面
    # 设置窗口大小、样式和布局，添加按钮、标签和进度条等控件
    def initUI(self):
        self.setWindowTitle("文件处理工具")
        self.setGeometry(150, 150, 300, 300)
        self.setStyleSheet("""
            QPushButton {
                background-color: rgba(59, 156, 156, 180);  /* 海蓝 + 半透明 */
                color: white;
                font-weight: bold;
                padding: 6px 12px;
                border: none;
                border-radius: 6px;
            }

            QPushButton:hover {
                background-color: rgba(59, 156, 156, 220);
            }

            QProgressBar {
                background-color: rgba(220, 240, 240, 100);  /* 浅灰蓝背景 */
                border: 1px solid #3B9C9C;
                border-radius: 10px;
                height: 18px;
                text-align: center;
                color: black;
            }

            QProgressBar::chunk {
                background-color: rgba(59, 156, 156, 180);  /* 海蓝进度块 */
                border-radius: 10px;
            }
        """)
        layout = QVBoxLayout()
        # 替代 layout.addLayout(folder_layout)



        # 输入输出文件夹标签
        folder_layout = QHBoxLayout()
        self.input_label = QLabel("输入文件夹: 未选择")
        self.output_label = QLabel("输出文件夹: 未选择")
        folder_layout = QVBoxLayout()
        folder_layout.addWidget(self.input_label)
        folder_layout.addWidget(self.output_label)

        button_layout = QHBoxLayout()
        self.input_button = QPushButton("选择输入文件夹")
        self.input_button.clicked.connect(self.select_input_folder)
        self.output_button = QPushButton("选择输出文件夹")
        self.output_button.clicked.connect(self.select_output_folder)
        button_layout.addWidget(self.input_button)
        button_layout.addWidget(self.output_button)
        folder_layout.addLayout(button_layout)

        self.folder_group = QGroupBox("📂 文件夹选项")
        self.folder_group.setLayout(folder_layout)
        layout.addWidget(self.folder_group)

        # 处理按钮
        self.process_button = QPushButton("➡️开始处理")
        self.process_button.clicked.connect(self.start_processing)
        layout.addWidget(self.process_button)

        # 退出按钮
        exit_button = QPushButton("⏏️退出程序")
        exit_button.clicked.connect(self.close)
        layout.addWidget(exit_button)

        self.help_button = QPushButton("📃帮助")
        self.help_button.clicked.connect(self.show_help)
        layout.addWidget(self.help_button)  # 把它添加到合适的布局中

        # 日志文本区
        # 日志+进度条组合区域
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)

        self.progress_bar = QProgressBar()

        self.log_group = QGroupBox("📄 日志记录")
        log_layout = QVBoxLayout()
        log_layout.addWidget(self.progress_bar)  # ✅ 进度条放进日志区域
        log_layout.addWidget(self.log_text)
        self.log_group.setLayout(log_layout)

        self.log_group.setVisible(False)  # 默认隐藏日志+进度条区域

        layout.addWidget(self.log_group)

        # 设置中心窗口
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # 初始化路径变量
        self.input_folder = None
        self.output_folder = None

    def show_help(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("帮助文档")
        dialog.resize(600, 500)

        layout = QVBoxLayout(dialog)
        text_browser = QTextBrowser()

        help_html = """
        <!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>生物医药数据处理集成工具使用说明书</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #2980b9;
            margin-top: 30px;
        }
        h3 {
            color: #3498db;
        }
        .toc {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
        }
        .toc h2 {
            margin-top: 0;
        }
        .toc ul {
            list-style-type: none;
            padding-left: 20px;
        }
        .toc li {
            margin-bottom: 8px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px 12px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        .indent {
            margin-left: 20px;
        }
        .question {
            font-weight: bold;
            color: #2c3e50;
        }
    </style>
</head>
<body>
    <h1>生物医药数据处理集成工具使用说明书</h1>
    
    <div class="toc">
        <h2>目录</h2>
        <ul>
            <li><a href="#overview">1. 软件概述</a></li>
            <li><a href="#installation">2. 安装与启动</a></li>
            <li><a href="#interface">3. 界面介绍</a></li>
            <li><a href="#workflow">4. 基本操作流程</a></li>
            <li><a href="#filetypes">5. 支持的文件类型</a></li>
            <li><a href="#modules">6. 处理模块详解</a>
                <ul>
                    <li><a href="#teg">TEG 血栓弹力图数据</a></li>
                    <li><a href="#aa">血小板聚集仪(AA)数据</a></li>
                    <li><a href="#transwell">Transwell 细胞穿膜实验</a></li>
                    <li><a href="#video">视频处理功能</a></li>
                    <li><a href="#molecule">分子结构文件转换</a></li>
                    <li><a href="#enzyme">酶标仪数据处理</a></li>
                    <li><a href="#gromacs">GROMACS XVG文件转换</a></li>
                </ul>
            </li>
            <li><a href="#output">7. 输出文件说明</a></li>
            <li><a href="#faq">8. 常见问题与解决方案</a></li>
            <li><a href="#contact">9. 联系与支持</a></li>
        </ul>
    </div>
    
    <section id="overview">
        <h2>1. 软件概述</h2>
        <p>本软件是为生物医药实验研究领域设计的数据处理集成工具，旨在简化实验数据的处理流程，提高数据处理效率和标准化程度。软件支持多种实验类型的数据处理，包括血栓弹力图(TEG)、血小板聚集分析、细胞穿膜实验、视频处理等。</p>
        <p>该工具采用图形化界面设计，无需编程知识即可操作，适合实验室科研人员和技术人员使用。处理后的数据以标准化格式输出，便于后续的统计分析和图表制作。</p>
    </section>
    
    <section id="installation">
        <h2>2. 安装与启动</h2>
        <h3>系统要求</h3>
        <ul>
            <li>Windows 7/8/10/11 操作系统</li>
            <li>至少 4GB RAM</li>
            <li>至少 500MB 可用磁盘空间</li>
        </ul>
        
        <h3>安装步骤</h3>
        <ol>
            <li>下载软件安装包(.exe文件)</li>
            <li>双击安装包文件运行安装程序</li>
            <li>按照安装向导的提示完成安装</li>
        </ol>
        
        <h3>启动软件</h3>
        <ul>
            <li>从桌面快捷方式或开始菜单中启动软件</li>
            <li>双击应用程序图标即可启动主界面</li>
        </ul>
    </section>
    
    <section id="interface">
        <h2>3. 界面介绍</h2>
        <p>软件启动后显示的主界面包含以下关键元素：</p>
        <ul>
            <li>文件夹设置区域：包含输入和输出文件夹的选择按钮和路径显示</li>
            <li>处理按钮：开始执行数据处理的主按钮</li>
            <li>退出按钮：关闭程序</li>
            <li>日志区域：显示处理过程中的状态信息和错误提示</li>
            <li>进度条：显示当前处理进度</li>
        </ul>
    </section>
    
    <section id="workflow">
        <h2>4. 基本操作流程</h2>
        <ol>
            <li>选择输入文件夹：点击"选择输入文件夹"按钮，选择包含待处理数据文件的文件夹</li>
            <li>选择输出文件夹：点击"选择输出文件夹"按钮，选择处理后文件的保存位置</li>
            <li>开始处理：点击"开始处理"按钮启动自动处理流程</li>
            <li>查看处理进度：通过进度条和日志信息跟踪处理状态</li>
            <li>查看结果：处理完成后，在指定的输出文件夹中查看生成的文件</li>
            <li>查看处理报告：处理完成后，会弹出处理结果统计窗口</li>
        </ol>
    </section>
    
    <section id="filetypes">
        <h2>5. 支持的文件类型</h2>
        <p>本软件能够自动识别并处理以下类型的文件：</p>
        <table>
            <tr>
                <th>文件扩展名</th>
                <th>数据类型</th>
                <th>处理方式</th>
            </tr>
            <tr>
                <td>.txt</td>
                <td>TEG血栓弹力图数据</td>
                <td>转换为CSV格式并进行可视化处理</td>
            </tr>
            <tr>
                <td>.xlsx, .xls, .xlsm</td>
                <td>血小板聚集仪(AA)数据或酶标仪数据</td>
                <td>根据数据特征进行识别和相应处理</td>
            </tr>
            <tr>
                <td>.tif, .jpg, .jpeg</td>
                <td>Transwell细胞穿膜实验图像</td>
                <td>分析紫色区域占比并生成结果文件</td>
            </tr>
            <tr>
                <td>.avi</td>
                <td>视频文件</td>
                <td>转换为MP4格式</td>
            </tr>
            <tr>
                <td>.mov, .mp4</td>
                <td>视频文件</td>
                <td>提取关键帧并进行荧光强度分析</td>
            </tr>
            <tr>
                <td>.cif</td>
                <td>分子结构文件</td>
                <td>转换为PDB格式</td>
            </tr>
            <tr>
                <td>.xvg</td>
                <td>GROMACS分子动力学模拟输出</td>
                <td>转换为CSV格式</td>
            </tr>
            <tr>
                <td>.csv</td>
                <td>多种类型数据</td>
                <td>根据内容特征进行识别和处理</td>
            </tr>
        </table>
    </section>
    
    <section id="modules">
        <h2>6. 处理模块详解</h2>
        
        <section id="teg">
            <h3>TEG 血栓弹力图数据</h3>
            <p><strong>支持文件类型：</strong>.txt</p>
            <p><strong>处理流程：</strong></p>
            <ol>
                <li>读取原始TEG数据文件</li>
                <li>提取x和y坐标数据</li>
                <li>添加z轴数据(y轴数据的负值)</li>
                <li>转换为标准CSV格式</li>
                <li>在可视化步骤中合并同一文件夹内的多个TEG文件</li>
            </ol>
            <p><strong>输出文件：</strong></p>
            <ul>
                <li>单个文件处理结果：output-[原文件名]_teg.csv</li>
                <li>文件夹汇总结果：visualized-[文件夹名]_teg.csv</li>
            </ul>
        </section>
        
        <section id="aa">
            <h3>血小板聚集仪(AA)数据</h3>
            <p><strong>支持文件类型：</strong>.xlsx, .xls, .xlsm, .csv</p>
            <p><strong>识别特征：</strong></p>
            <ul>
                <li>第一行值为"NjData"</li>
                <li>第二行值为"ADPrateData"</li>
            </ul>
            <p><strong>处理流程：</strong></p>
            <ol>
                <li>识别特定格式的血小板聚集仪数据</li>
                <li>移除前两行标题信息</li>
                <li>处理特殊分隔符"@#"</li>
                <li>转置数据结构</li>
                <li>输出为标准Excel格式</li>
            </ol>
            <p><strong>输出文件：</strong>output-[原文件名].xlsx</p>
        </section>
        
        <section id="transwell">
            <h3>Transwell 细胞穿膜实验</h3>
            <p><strong>支持文件类型：</strong>.tif, .jpg, .jpeg</p>
            <p><strong>处理流程：</strong></p>
            <ol>
                <li>读取图像文件</li>
                <li>转换为HSV颜色空间</li>
                <li>检测并计算紫色区域占比</li>
                <li>生成包含文件名和紫色占比的CSV文件</li>
                <li>汇总同一文件夹内的所有Transwell结果</li>
            </ol>
            <p><strong>输出文件：</strong></p>
            <ul>
                <li>单个文件处理结果：output-[原文件名]_transwell.csv</li>
                <li>文件夹汇总结果：visualized-[文件夹名]_transwell.csv</li>
                <li>全局汇总结果：summarized-transwell.csv</li>
            </ul>
        </section>
        
        <section id="video">
            <h3>视频处理功能</h3>
            
            <h4>视频格式转换：</h4>
            <ul>
                <li><strong>支持文件类型：</strong>.avi</li>
                <li><strong>处理方式：</strong>转换为MP4格式</li>
                <li><strong>输出文件：</strong>output-[原文件名]_a2m.mp4</li>
            </ul>
            
            <h4>荧光强度分析(FA)：</h4>
            <ul>
                <li><strong>支持文件类型：</strong>.mov, .mp4 (小于10分钟的视频)</li>
                <li><strong>处理流程：</strong></li>
                <ol>
                    <li>自动检测视频时长</li>
                    <li>对视频中五个固定区域(左上、右上、左下、右下、中心)进行荧光强度分析</li>
                    <li>每秒采样一次，计算平均强度</li>
                    <li>生成时间序列数据</li>
                </ol>
                <li><strong>输出文件：</strong>output-[原文件名]_fa.csv</li>
            </ul>
            
            <h4>视频截图提取：</h4>
            <ul>
                <li><strong>支持文件类型：</strong>.mov, .mp4</li>
                <li><strong>处理流程：</strong></li>
                <ol>
                    <li>根据视频时长确定时间点：
                        <ul>
                            <li>小于5分钟：1, 2, 3分钟</li>
                            <li>5-10分钟：1, 3, 5分钟</li>
                            <li>大于10分钟：3, 6, 9, 12, 15分钟</li>
                        </ul>
                    </li>
                    <li>在指定时间点提取视频帧</li>
                    <li>保存为JPEG图像</li>
                </ol>
                <li><strong>输出文件：</strong>保存在以原视频名命名的子文件夹中，格式为[视频名]_[分钟]min.jpg</li>
            </ul>
        </section>
        
        <section id="molecule">
            <h3>分子结构文件转换</h3>
            <p><strong>支持文件类型：</strong>.cif</p>
            <p><strong>处理流程：</strong></p>
            <ol>
                <li>使用Open Babel工具将CIF格式转换为PDB格式</li>
                <li>保留原始文件的分子结构信息</li>
            </ol>
            <p><strong>输出文件：</strong>output-[原文件名].pdb</p>
        </section>
        
        <section id="enzyme">
            <h3>酶标仪数据处理</h3>
            <p><strong>支持文件类型：</strong>.xlsx, .xls, .xlsm, .csv</p>
            <p><strong>处理流程：</strong></p>
            <ol>
                <li>读取96孔板格式的原始数据</li>
                <li>重组数据结构为标准表格形式</li>
                <li>适当处理行列结构，便于后续分析</li>
            </ol>
            <p><strong>输出文件：</strong>output-[原文件名].xlsx</p>
        </section>
        
        <section id="gromacs">
            <h3>GROMACS XVG文件转换</h3>
            <p><strong>支持文件类型：</strong>.xvg</p>
            <p><strong>处理流程：</strong></p>
            <ol>
                <li>读取XVG文件内容</li>
                <li>过滤掉以@或#开头的注释行</li>
                <li>提取数值数据</li>
                <li>转换为CSV格式</li>
            </ol>
            <p><strong>输出文件：</strong>保存在"xvg_csv"子文件夹中，格式为[当前文件夹名]_[原文件名].csv</p>
        </section>
    </section>
    
    <section id="output">
        <h2>7. 输出文件说明</h2>
        <p>处理后的文件会按照以下规则命名并保存：</p>
        <ol>
            <li><strong>单个文件处理结果：</strong>前缀为"output-"，后跟原文件名和处理类型标识
                <ul>
                    <li>例如：output-sample_teg.csv、output-experiment_fa.csv</li>
                </ul>
            </li>
            <li><strong>可视化汇总文件：</strong>前缀为"visualized-"，后跟文件夹名和处理类型标识
                <ul>
                    <li>例如：visualized-experiment1_teg.csv、visualized-test2_transwell.csv</li>
                </ul>
            </li>
            <li><strong>全局汇总文件：</strong>前缀为"summarized-"，后跟处理类型标识
                <ul>
                    <li>例如：summarized-transwell.csv</li>
                </ul>
            </li>
            <li><strong>特殊处理文件：</strong>
                <ul>
                    <li>视频截图：保存在以原视频名命名的子文件夹中</li>
                    <li>XVG转换结果：保存在"xvg_csv"子文件夹中</li>
                </ul>
            </li>
        </ol>
    </section>
    
    <section id="faq">
        <h2>8. 常见问题与解决方案</h2>
        <p class="question">Q: 软件无法识别我的文件类型</p>
        <p>A: 请确认文件扩展名正确，并且文件内容符合相应格式要求。对于特殊格式的文件，可能需要预先进行格式转换。</p>
        
        <p class="question">Q: 处理过程中出现错误</p>
        <p>A: 查看日志区域的错误信息，常见原因包括：文件格式不正确、文件损坏、或缺少必要的数据列。尝试检查原始文件并修复后重新处理。</p>
        
        <p class="question">Q: 输出文件中数据不完整</p>
        <p>A: 可能是原始数据中包含异常值或格式不一致。检查原始文件，确保数据格式符合处理要求。</p>
        
        <p class="question">Q: 视频处理速度较慢</p>
        <p>A: 视频处理通常较为耗时，特别是高分辨率或长时间的视频。请耐心等待处理完成。</p>
        
        <p class="question">Q: Open Babel相关错误</p>
        <p>A: 分子结构文件转换依赖于Open Babel工具。请确保系统中正确安装了Open Babel，并已添加到系统PATH中。</p>
    </section>
    
    <section id="contact">
        <h2>9. 联系与支持</h2>
        <p>如遇到无法解决的问题，或有功能改进建议，请联系：</p>
        <p><strong>作者：</strong>王天宇，周绍芸，申传斌</p>
        <p><strong>邮箱：</strong>Terrywangtianyu@gmail.com</p>
        <p><strong>单位：</strong>中国海洋大学血液与心脑血管药理课题组 (OUC Blood and Cardiovascular Pharmacology Research Group)</p>
    </section>
</body>
</html>
        """

        text_browser.setHtml(help_html)
        layout.addWidget(text_browser)

        dialog.setLayout(layout)
        dialog.exec()


    # FileProcessorApp.select_input_folder 方法
    # 选择输入文件夹的方法
    # 打开文件选择对话框，并更新界面上的输入文件夹路径显示
    def select_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择输入文件夹")
        if folder:
            self.input_folder = folder
            self.input_label.setText(f"输入文件夹: {folder}")

    # FileProcessorApp.select_output_folder 方法
    # 选择输出文件夹的方法
    # 打开文件选择对话框，并更新界面上的输出文件夹路径显示
    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if folder:
            self.output_folder = folder
            self.output_label.setText(f"输出文件夹: {folder}")

    # FileProcessorApp.start_processing 方法
    # 开始处理文件的方法
    # 调整UI显示状态，创建并启动文件处理线程
    def start_processing(self):
        if not self.input_folder or not self.output_folder:
            self.log_text.append("请先选择输入和输出文件夹!")
            return

        # 重置界面状态
        self.progress_bar.setValue(0)
        self.log_text.clear()
        self.log_group.setVisible(True)
        self.folder_group.setVisible(False)
        self.input_button.setVisible(False)
        self.output_button.setVisible(False)
        self.process_button.setVisible(False)

        # ✅ 创建处理线程
        self.processing_thread = FileProcessorThread(self.input_folder, self.output_folder)
        self.processing_thread.update_progress.connect(self.update_progress)
        self.processing_thread.update_log.connect(self.update_log)
        self.processing_thread.processing_completed.connect(self.processing_completed)
        self.processing_thread.finished.connect(self.on_processing_thread_finished)  # ✅ 正确连接 finished 信号

        # ✅ 启动线程
        self.processing_thread.start()

    # FileProcessorApp.update_progress 方法
    # 更新进度条显示的方法
    # 接收处理线程发送的进度信号并更新UI
    def update_progress(self, value):
        self.progress_bar.setValue(value)

    # FileProcessorApp.update_log 方法
    # 更新日志文本的方法
    # 接收处理线程发送的日志信息并显示到界面
    def update_log(self, message):
        self.log_text.append(message)

    # FileProcessorApp.processing_completed 方法
    # 处理完成后的回调方法
    # 显示处理统计信息，包括各类文件数量和错误信息，并弹出结果对话框
    def processing_completed(self, processed_files, file_type_counts, error_files):
        summary_text = "处理完成！文件处理统计：\n"
        summary_text += f"    总文件数：{file_type_counts['Total']}\n"
        summary_text += f"    TEG文件：{file_type_counts['TEG']}\n"
        summary_text += f"    血小板聚集仪文件：{file_type_counts['AA']}\n"
        summary_text += f"    Transwell文件：{file_type_counts['Transwell']}\n"
        summary_text += f"    AVI转MP4文件：{file_type_counts['AVI2MP4']}\n"
        summary_text += f"    PerfusionChamber文件：{file_type_counts['MOV_MP4']}\n"
        summary_text += f"    复钙文件：{file_type_counts['Excel_CSV']}\n"
        summary_text += f"    cif转pdb文件：{file_type_counts['cif2pdb']}\n"
        summary_text += f"    xvg转CSV文件：{file_type_counts['xvg2csv']}\n"
        summary_text += f"    video2pic 文件：{file_type_counts['video2pic']}\n"

        # 添加错误文件报告
        if error_files:
            summary_text += "\n出错文件列表：\n"
            for error_file in error_files:
                summary_text += f"文件: {error_file['file']}\n"
                summary_text += f"错误信息: {error_file['error_message']}\n\n"

# 在原有日志追加的基础上，添加弹窗
        self.log_text.append(summary_text)

        # 添加弹窗提醒
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.setText("FINISHED")
        msg_box.setDetailedText(summary_text)
        msg_box.setWindowTitle("RESULT")
        msg_box.exec()

    def on_processing_thread_finished(self):
        self.log_text.append("✅ 所有处理完成！可以在输出文件夹中查看结果。")

        # 如果你希望处理完成后恢复按钮可点击，也可以添加：
        self.folder_group.setVisible(True)
        self.input_button.setVisible(True)
        self.output_button.setVisible(True)
        self.process_button.setVisible(True)


# process_teg 函数
# 处理TEG(血栓弹力图)数据文件，将输入的txt文件转换为带x、y、z三列的csv文件
# 其中z列是y列的负值，用于数据可视化
def process_teg(file_path, output_file_path):
    # 读取 CSV 文件并设置列名
    df = pd.read_csv(file_path, header=None, names=['x', 'y'])

    # 替换非数字和小数点的字符为空字符串
    df = df.replace({r'[^0-9.]': ''}, regex=True)

    # 新增一列 'z'，值为 'y' 列加负号
    df['z'] = '-' + df['y']

    # 保存处理后的数据到指定输出文件
    df.to_csv(output_file_path, index=False)

    # 可选：打印处理完成的提示
    print(f"处理 CSV 文件: {output_file_path}")
# process_video_screenshots 函数
# 在视频的指定时间点截取帧并保存为图片
# 根据传入的时间点列表(单位:分钟)生成截图，并保存到以视频名称命名的子文件夹中
def process_video_screenshots(video_path, output_folder, timestamps):
    """
    对单个视频文件，在指定的时间点（以分钟为单位）生成截图，
    截图保存于 output_folder 下一个以视频文件名命名的子文件夹中。

    参数：
        video_path: str
            视频文件的完整路径。
        output_folder: str
            指定的输出根目录。
        timestamps: list of int/float
            时间点列表（单位分钟），将在这些时间点处截取视频帧。
    """

    # 获取视频文件基本名称（不含扩展名）
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # 构建该视频专属的输出文件夹路径
    video_output_folder = os.path.join(output_folder, video_name)
    os.makedirs(video_output_folder, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps == 0:
        print(f"无法获取视频 {video_path} 的帧率！")
        cap.release()
        return

    # 遍历指定的时间点，生成截图
    for minute in timestamps:
        # 计算第 minute 分钟对应的帧号
        frame_number = int(fps * 60 * minute)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            # 构造截图图片名称，格式为 "视频名_分钟min.jpg"
            img_filename = f"{video_name}_{minute}min.jpg"
            img_path = os.path.join(video_output_folder, img_filename)
            cv2.imwrite(img_path, frame)
        else:
            # 此处仅打印错误，详细错误处理由主脚本记录日志或计数
            print(f"视频 {video_path} 在 {minute} 分钟处截图失败！")
    cap.release()

# process_cif2pdb 函数
# 将.cif分子结构文件转换为.pdb格式
# 调用Open Babel工具执行转换，适用于分子建模数据处理
def process_cif2pdb(file_path, output_file_path):
    """
    将指定的 .cif 文件转换为 .pdb 文件。依赖于 Open Babel 工具，
    要求传入的文件名中包含 'model_0' 作为符合条件的标识。
    """
    try:
        subprocess.run(["obabel", file_path, "-O", output_file_path], check=True)
        print(f"Converted {file_path} to {output_file_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting {file_path}: {e}")

# visualize_teg_files 函数
# 收集并整合所有TEG数据文件，生成可视化汇总文件
# 递归处理所有子文件夹中的TEG文件，汇总成单个可视化CSV文件便于绘图分析
def visualize_teg_files(input_folder, output_folder):
    print(f"开始TEG可视化处理: {input_folder}")

    # 存储所有生成的可视化文件路径
    all_visualized_files = []

    # 递归处理所有子文件夹
    for root, dirs, files in os.walk(input_folder):
        # 计算当前处理的输出文件夹
        relative_path = os.path.relpath(root, input_folder)
        current_output_folder = os.path.join(output_folder, relative_path)

        # 确保输出文件夹存在
        os.makedirs(current_output_folder, exist_ok=True)

        # 严格地筛选TEG文件
        teg_files = []
        for f in os.listdir(current_output_folder):
            if f.startswith('output-') and f.endswith('.csv'):
                file_path = os.path.join(current_output_folder, f)
                try:
                    # 读取文件并检查列
                    df = pd.read_csv(file_path)
                    # 严格检查是否包含必需的列，且数据符合预期
                    if set(['x', 'y', 'z']).issubset(df.columns) :
                        teg_files.append(f)
                except Exception as e:
                    print(f"校验文件 {f} 时出错: {e}")

        teg_files = natsort.natsorted(teg_files)  # 自然排序文件名
        print(f"检测到TEG文件: {teg_files}")

        # 如果没有文件，跳过当前文件夹
        if not teg_files:
            continue

        # 存储所有数据
        all_data = {}
        max_x_length = 0

        # 读取所有文件
        for file in teg_files:
            file_path = os.path.join(current_output_folder, file)
            try:
                df = pd.read_csv(file_path)

                # 记录最长x轴长度
                max_x_length = max(max_x_length, len(df))

                # 使用文件名作为列名存储y和z数据
                all_data[file] = {
                    'y': df['y'].tolist(),
                    'z': df['z'].tolist()
                }
            except Exception as e:
                print(f"读取文件 {file} 时出错: {e}")

        # 如果没有成功读取任何数据
        if not all_data:
            print("未成功读取任何TEG数据")
            continue

        # 创建统一的x轴
        x_axis = [x * 5 for x in range(max_x_length)]

        # 创建最终的DataFrame
        result_df = pd.DataFrame({'x': x_axis})

        # 填充数据，缺失值用NaN补全
        for filename, data in all_data.items():
            result_df[f'{filename}_y'] = pd.Series(data['y'] + [None] * (max_x_length - len(data['y'])))
            result_df[f'{filename}_z'] = pd.Series(data['z'] + [None] * (max_x_length - len(data['z'])))

        # 生成可视化CSV文件
        visualized_file_path = os.path.join(current_output_folder, f'visualized-{os.path.basename(root)}_teg.csv')
        result_df.to_csv(visualized_file_path, index=False)
        print(f"生成可视化CSV: {visualized_file_path}")

        # 收集所有生成的可视化文件路径
        all_visualized_files.append(visualized_file_path)

    return all_visualized_files  # 返回所有生成的文件路径列表
# process_mr 函数
# 处理酶标仪(MicroReader)数据文件，转换为标准格式
# 将96孔板格式数据重组为更易于分析的表格形式
def process_mr(file_path, output_file_path):
    # 1. 读取数据
    # skiprows=1 跳过第一行标题（Reading 1）
    # header=None 方便后续通过索引切片
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, header=None, skiprows=1, encoding='GB18030')
    else:
        df = pd.read_excel(file_path, header=None, skiprows=1)

    # 2. 提取纯数据区域 (第 1 到 12 列)
    # 第一列 (索引 0) 是 A, B, C... 标签，予以排除
    data_only = df.iloc[:, 1:13]

    # 3. 准备容器和标签
    all_plates = []
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    numbers = [str(i) for i in range(1, 13)]
    well_labels = [f'{l}{n}' for l in letters for n in numbers]  # 生成 A1, A2... H12

    i = 0
    while True:
        # 假设每块板 8 行，板间有 1 个空行（跨度为 9）
        # 如果你的文件里板子是紧挨着的，请将 9 改为 8
        start_row = i * 9
        table = data_only.iloc[start_row: start_row + 8, :]

        # 检查是否还有数据
        if table.empty or len(table) < 8:
            break

        # 4. 关键点：将 8x12 压平为一行
        # .stack().values 会按 A1, A2, A3... 的顺序生成一个 1D 数组
        flattened_row = table.stack().values
        all_plates.append(flattened_row)
        i += 1

    # 5. 构建最终 DataFrame
    # 将列表转换为 DataFrame，并以孔位作为列名
    result = pd.DataFrame(all_plates, columns=well_labels)

    # 可选：插入一列显示这是第几次读数
    result.insert(0, 'Reading_Index', [f'Reading_{j + 1}' for j in range(len(result))])

    # 6. 保存
    if file_path.endswith('.csv'):
        result.to_csv(output_file_path, index=False)
    else:
        result.to_excel(output_file_path, index=False)

    print(f"处理完成！列标题为孔位，每一行对应一次读数。保存至: {output_file_path}")


# xvg2csv 函数
# 将GROMACS的.xvg分子动力学模拟输出文件转换为CSV格式
# 忽略以@或#开头的注释行，仅保留数值数据
def xvg2csv(input_folder, output_dir):
    """
    遍历指定 input_folder 目录下的所有 .xvg 文件（不递归子文件夹），
    忽略以 @ 或 # 开头的注释行，将数据转换为 CSV 文件，
    输出到 output_dir 目录中，输出文件名格式为：
    {当前文件夹名}_{原文件名}.csv
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取当前目录名作为文件名前缀
    current_folder_name = os.path.basename(os.path.normpath(input_folder))

    # 列出指定目录中的所有文件（不使用 os.walk 递归）
    for file in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file)
        if os.path.isfile(file_path) and file.endswith(".xvg"):
            # 读取 .xvg 文件（忽略注释行）
            data = []
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith(('#', '@')):
                        try:
                            parts = line.split()
                            data.append(parts)
                        except Exception as e:
                            print(f"Error parsing line in {file_path}: {line}")

            if not data:
                print(f"Warning: No data in {file_path}")
                continue

            # 转换数据为 DataFrame
            df = pd.DataFrame(data)

            # 构造输出文件名：当前目录名 + "_" + 原文件名（不含扩展名） + ".csv"
            csv_name = f"{current_folder_name}_{os.path.splitext(file)[0]}.csv"
            csv_path = os.path.join(output_dir, csv_name)

            # 写入 CSV 文件，不包含索引与表头
            df.to_csv(csv_path, index=False, header=False)
            print(f"Converted: {file_path} → {csv_path}")

# process_fa 函数
# 处理血流灌注(Flow Adhesion)视频，分析五个固定区域的荧光强度
# 每秒采样一次，计算左上、右上、左下、右下、中心五个区域的平均强度值
def process_fa(video_path, output_file_path):
    def get_quadrants(frame):
        height, width = frame.shape[:2]
        mid_x = width // 2
        mid_y = height // 2
        quadrants = [
            (0, 0, mid_x, mid_y),                  # 左上
            (mid_x, 0, width - mid_x, mid_y),      # 右上
            (0, mid_y, mid_x, height - mid_y),     # 左下
            (mid_x, mid_y, width - mid_x, height - mid_y)  # 右下
        ]
        return quadrants

    def analyze_frame(frame, regions, scale_factor=10000000 / 255):
        frame_results = []
        for x, y, w, h in regions:
            roi = frame[y:y + h, x:x + w]
            mean_intensity = roi.mean()
            frame_results.append(mean_intensity * scale_factor)
        return frame_results

    def replace_outliers(data, threshold_factor=3):
        """基于相邻差值的离群点替换"""
        import numpy as np
        data = np.array(data, dtype=float)
        if len(data) < 3:
            return data.tolist()
        diffs = np.diff(data)
        threshold = threshold_factor * np.std(diffs)
        for i in range(1, len(data) - 1):
            if abs(data[i] - data[i-1]) > threshold and abs(data[i] - data[i+1]) > threshold:
                data[i] = (data[i-1] + data[i+1]) / 2
        return data.tolist()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    ret, first_frame = cap.read()
    if not ret:
        print(f"无法读取视频: {video_path}")
        return

    first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    quadrants = get_quadrants(first_frame_gray)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    output_file_path_csv = output_file_path.replace('.xlsx', '.csv')
    results_buffer = []  # 缓存所有结果，便于后续整体去异常值

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_count % int(fps) == 0:
            frame_results = analyze_frame(gray_frame, quadrants)
            results_buffer.append(frame_results)

        frame_count += 1

    cap.release()

    # 对每一列单独进行离群值替换
    import numpy as np
    results_array = np.array(results_buffer)
    for col in range(results_array.shape[1]):
        results_array[:, col] = replace_outliers(results_array[:, col])

    # 保存到 CSV
    import csv
    with open(output_file_path_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['time(sec)', 'top_left', 'top_right', 'bottom_left', 'bottom_right'])
        for t, row in enumerate(results_array):
            writer.writerow([t] + row.tolist())

    print(f"完成视频分析: {video_path}")


# --- reserved for extension ---
def future_extension_hook():
    pass

def not_implemented_yet():
    print("This feature is not yet implemented.")

def temp_dev_function():
    # developer testing placeholder
    dummy = [i for i in range(10)]
    return dummy

def debug_placeholder(*args, **kwargs):
    print("Debug placeholder hit with args:", args, "and kwargs:", kwargs)
# --- reserved for extension ---


def visualize_fa_files(input_folder, output_folder):
    print(f"开始FA可视化处理: {input_folder}")

    # 存储所有生成的可视化文件路径
    all_visualized_files = []

    # 递归处理所有子文件夹
    for root, dirs, files in os.walk(input_folder):
        # 计算当前处理的输出文件夹
        relative_path = os.path.relpath(root, input_folder)
        current_output_folder = os.path.join(output_folder, relative_path)

        # 确保输出文件夹存在
        os.makedirs(current_output_folder, exist_ok=True)

        # 筛选FA文件
        fa_files = []
        for f in os.listdir(current_output_folder):
            if f.startswith('output') and f.endswith('fa.csv'):
                file_path = os.path.join(current_output_folder, f)
                try:
                    # 读取文件并检查列
                    df = pd.read_csv(file_path)
                    # 检查是否为FA处理后的文件（包含time和5个区域数据）
                    if len(df.columns) == 6 and 'time(sec)' in df.columns:
                        fa_files.append(f)
                except Exception as e:
                    print(f"校验文件 {f} 时出错: {e}")
                    continue

        fa_files = natsort.natsorted(fa_files)  # 自然排序文件名
        print(f"检测到FA文件: {fa_files}")

        # 如果没有文件，跳过当前文件夹
        if not fa_files:
            continue

        # 存储所有数据
        all_data = {}
        max_time_length = 0

        # 读取所有文件
        for file in fa_files:
            file_path = os.path.join(current_output_folder, file)
            try:
                df = pd.read_csv(file_path)

                # 更新最长时间列
                if len(df) > max_time_length:
                    max_time_length = len(df)
                    time_column = df['time(sec)']

                # 存储除time列外的所有数据列
                data_columns = df.drop('time(sec)', axis=1)
                base_name = os.path.splitext(file)[0].replace('output-', '')

                # 为每个区域创建带文件名前缀的列名
                for col in data_columns.columns:
                    col_name = f'{base_name}_{col}'
                    all_data[col_name] = data_columns[col].tolist()

            except Exception as e:
                print(f"读取文件 {file} 时出错: {e}")

        # 如果没有成功读取任何数据
        if not all_data or time_column is None:
            print(f"未成功读取 {relative_path} 中的FA数据")
            continue

        # 创建最终的DataFrame
        result_df = pd.DataFrame({'time(sec)': time_column})

        # 填充数据，缺失值用NaN补全
        for col_name, data in all_data.items():
            result_df[col_name] = pd.Series(data + [None] * (max_time_length - len(data)))

        # 生成可视化CSV文件
        visualized_file_path = os.path.join(current_output_folder, f'visualized-{os.path.basename(root)}_fa.csv')
        result_df.to_csv(visualized_file_path, index=False)
        print(f"生成可视化CSV: {visualized_file_path}")

        # 收集所有生成的可视化文件路径
        all_visualized_files.append(visualized_file_path)

    return all_visualized_files
# process_aa 函数
# 处理血小板聚集仪(Aggregation Analyzer)数据
# 解析特殊格式的血小板聚集数据，转换为Excel格式便于分析
def process_aa(file_path, output_file_path):
    # 读取输入文件
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, header=None)
    else:
        df = pd.read_excel(file_path, header=None)

    # 移除前两行
    df = df.iloc[2:]

    # 处理数据分割
    processed_data = []
    for row in df.values:
        processed_row = []
        for cell in row:
            # 确保是字符串且包含'@#'
            if isinstance(cell, str) and '@#' in cell:
                processed_row.extend(cell.split('@#'))
            elif pd.notna(cell):
                processed_row.append(str(cell))

        if processed_row:  # 只添加非空行
            processed_data.append(processed_row)

    # 转置数据
    if processed_data:
        transposed_data = list(map(list, zip(*processed_data)))
    else:
        transposed_data = []

    # 创建并保存Excel
    wb = openpyxl.Workbook()
    sheet = wb.active

    for row in transposed_data:
        sheet.append(row)

    wb.save(output_file_path)
    print(f"Analyzing LTA files: {output_file_path}")


# process_avi2mp4 函数
# 将AVI格式视频转换为MP4格式
# 读取所有帧然后重新以MP4编码写入，保持原始分辨率和帧率
def process_avi2mp4(videoPath, outVideoPath):
    capture = cv2.VideoCapture(videoPath)
    fps = capture.get(cv2.CAP_PROP_FPS)  # 获取帧率
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # fNUMS = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    suc = capture.isOpened()  # 是否成功打开

    allFrame = []
    while suc:
        suc, frame = capture.read()
        if suc:
            allFrame.append(frame)
    capture.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    videoWriter = cv2.VideoWriter(outVideoPath, fourcc, fps, size)
    for aFrame in allFrame:
        videoWriter.write(aFrame)
    videoWriter.release()
    print(f"Analyzing video file: {outVideoPath}")
# process_transwell 函数
# 处理细胞穿膜(Transwell)实验的图像数据
# 通过HSV颜色空间检测图像中紫色区域，计算穿膜细胞占比
def process_transwell(file_path, output_file_path):
    # 读取图像
    image = cv2.imread(file_path)

    # 转换为HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义紫色HSV范围（可能需要根据具体图像调整）
    lower_purple = np.array([120, 50, 50])  # 紫色的下边界
    upper_purple = np.array([160, 255, 255])  # 紫色的上边界

    # 创建紫色区域的掩码
    purple_mask = cv2.inRange(hsv_image, lower_purple, upper_purple)

    # 计算紫色区域占比
    total_pixels = purple_mask.size
    purple_pixels = cv2.countNonZero(purple_mask)
    purple_percentage = (purple_pixels / total_pixels) * 100

    # 获取文件名（不含扩展名）
    filename = os.path.splitext(os.path.basename(file_path))[0]

    # 创建DataFrame并保存
    df = pd.DataFrame({
        'filename': [filename],
        'purple_percentage': [purple_percentage]
    })

    # 保存为CSV
    df.to_csv(output_file_path, index=False)

    print(f"处理Transwell图像文件: {output_file_path}")


# visualize_transwell_files 函数
# 收集并整合所有Transwell数据文件，生成可视化汇总文件
# 递归处理子文件夹中的Transwell分析结果，合并为单个可视化CSV文件
def visualize_transwell_files(input_folder, output_folder):
    print(f"开始Transwell可视化处理: {input_folder}")

    # 存储所有生成的可视化文件路径
    all_visualized_files = []

    # 递归处理所有子文件夹
    for root, dirs, files in os.walk(input_folder):
        # 计算当前处理的输出文件夹
        relative_path = os.path.relpath(root, input_folder)
        current_output_folder = os.path.join(output_folder, relative_path)

        # 确保输出文件夹存在
        os.makedirs(current_output_folder, exist_ok=True)

        # 更严格地筛选Transwell文件
        transwell_files = []
        for f in os.listdir(current_output_folder):
            if f.startswith('output-') and f.endswith('.csv'):
                file_path = os.path.join(current_output_folder, f)
                try:
                    # 读取文件并检查列
                    df = pd.read_csv(file_path)
                    # 严格检查是否为Transwell处理后的文件
                    if df.columns.tolist() == ['filename', 'purple_percentage']:
                        transwell_files.append(f)
                except Exception as e:
                    print(f"校验文件 {f} 时出错: {e}")
                    continue

        transwell_files = natsort.natsorted(transwell_files)  # 自然排序文件名
        print(f"检测到Transwell文件: {transwell_files}")

        # 如果没有文件，跳过当前文件夹
        if not transwell_files:
            continue

        # 读取并整合所有文件
        result_df = pd.DataFrame(columns=['filename', 'purple_percentage'])

        for file in transwell_files:
            file_path = os.path.join(current_output_folder, file)
            try:
                df = pd.read_csv(file_path)
                result_df = pd.concat([result_df, df], ignore_index=True)
            except Exception as e:
                print(f"读取文件 {file} 时出错: {e}")

        # 如果没有成功读取任何数据
        if result_df.empty:
            print(f"未成功读取 {relative_path} 中的Transwell数据")
            continue

        # 生成可视化CSV文件
        visualized_file_path = os.path.join(current_output_folder, f'visualized-{os.path.basename(root)}_transwell.csv')
        result_df.to_csv(visualized_file_path, index=False)
        print(f"生成可视化CSV: {visualized_file_path}")

        # 收集所有生成的可视化文件路径
        all_visualized_files.append(visualized_file_path)

    return all_visualized_files  # 返回所有生成的文件路径列表

# summarize_transwell_files 函数
# 生成所有Transwell实验结果的总结报告
# 收集所有可视化后的Transwell数据，添加文件夹路径信息，整合为单个总结文件
def summarize_transwell_files(output_folder):
    print(f"开始Transwell总结处理: {output_folder}")

    # 所有整合后的数据
    final_result_df = pd.DataFrame(columns=['folderpath', 'filename', 'purple_percentage'])

    # 遍历 output_folder 中的所有子文件夹
    for root, dirs, files in os.walk(output_folder):
        # 计算当前处理的文件夹的相对路径
        relative_path = os.path.relpath(root, output_folder)

        # 筛选 Transwell 可视化文件
        transwell_visualized_files = []
        for f in files:
            if f.startswith('visualized-') and f.endswith('_transwell.csv'):
                file_path = os.path.join(root, f)
                try:
                    # 读取文件并检查列
                    df = pd.read_csv(file_path)
                    # 检查是否包含严格匹配的列名
                    if set(df.columns) == {'filename', 'purple_percentage'}:
                        transwell_visualized_files.append(file_path)
                except Exception as e:
                    print(f"校验文件 {f} 时出错: {e}")
                    continue

        # 对文件名进行排序
        transwell_visualized_files = natsort.natsorted(transwell_visualized_files)
        print(f"检测到Transwell可视化文件: {transwell_visualized_files}")

        # 如果没有符合条件的文件，跳过当前文件夹
        if not transwell_visualized_files:
            continue

        # 读取并整合当前文件夹的所有文件
        folder_result_df = pd.DataFrame(columns=['filename', 'purple_percentage'])
        for file_path in transwell_visualized_files:
            try:
                df = pd.read_csv(file_path)
                folder_result_df = pd.concat([folder_result_df, df], ignore_index=True)
            except Exception as e:
                print(f"读取文件 {file_path} 时出错: {e}")

        # 如果没有成功读取任何数据
        if folder_result_df.empty:
            print(f"未成功读取 {relative_path} 中的Transwell数据")
            continue

        # 添加 folderpath 列
        folder_result_df['folderpath'] = relative_path

        # 重新排序列
        folder_result_df = folder_result_df[['folderpath', 'filename', 'purple_percentage']]

        # 追加到最终结果
        final_result_df = pd.concat([final_result_df, folder_result_df], ignore_index=True)

    # 如果没有任何数据
    if final_result_df.empty:
        print("未找到任何Transwell数据进行总结")
        return None

    # 生成总结 CSV 文件
    summary_file_path = os.path.join(output_folder, 'summarized-transwell.csv')
    final_result_df.to_csv(summary_file_path, index=False)
    print(f"生成Transwell总结文件: {summary_file_path}")

    return summary_file_path



# 顶部添加（在 __main__ 外）
main_window = None

if __name__ == "__main__":
    from PyQt6.QtGui import QPixmap
    from PyQt6.QtWidgets import QApplication, QSplashScreen
    from PyQt6.QtCore import Qt, QTimer

    def resource_path(relative_path):
        import sys, os
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, relative_path)
        return os.path.join(os.path.abspath("."), relative_path)

    app = QApplication(sys.argv)

    # 获取屏幕信息
    screen = app.primaryScreen()
    screen_size = screen.size()
    screen_width = screen_size.width()
    screen_height = screen_size.height()

    # 缩放 splash 图片
    original_pixmap = QPixmap(resource_path("newrocket.png"))
    scaled_pixmap = original_pixmap.scaled(screen_width // 2, screen_height // 2,
                                           Qt.AspectRatioMode.KeepAspectRatio,
                                           Qt.TransformationMode.SmoothTransformation)

    splash = QSplashScreen(scaled_pixmap, Qt.WindowType.WindowStaysOnTopHint)
    splash.setWindowFlags(Qt.WindowType.FramelessWindowHint)

    # 居中 splash
    splash_width = splash.size().width()
    splash_height = splash.size().height()
    splash.move((screen_width - splash_width) // 2, (screen_height - splash_height) // 2)

    splash.show()

    def start_app():
        global main_window
        splash.close()
        main_window = FileProcessorApp()
        main_window.show()

    QTimer.singleShot(2500, start_app)
    sys.exit(app.exec())
