web_source   这是前端原代码模板
website这是训练后保存的路径

linux 安装conda
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  安装

	bash Miniconda3-latest-Linux-x86_64.sh   运行
		直接按 Enter 键（使用默认路径 /root/miniconda3）。

		不要修改路径，除非你有特殊需求（如磁盘空间不足）。
		
		修改路径可能导致权限问题或后续配置错误。
		
		等待安装完成：
		安装过程会自动解压文件并设置环境，需耐心等待（约1~3分钟）。
    
激活conda 
	使用这个激活 
	echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
	source ~/.bashrc

conda create -n human python=3.11
conda activate human
pip install torch
pip install -r requirements.txt
conda install -c conda-forge ffmpeg


conda install -c conda-forge nodejs
npm install -g yarn
yarn add javascript-obfuscator terser
 
nohup conda run -n human python app.py > app.log 2>&1 &    运行

pkill -f "python app.py"  杀死所有的app.py
ps aux | grep "python app.py"  查看

conda deactivate  退出环境
 

本产品包含基于 LKZMuZiLi/human 的代码。  
LKZMuZiLi/human 的原始许可证如下：  
https://github.com/LKZMuZiLi/human

本产品包含基于 DH_live（MIT License）的代码。  
DH_live 的原始许可证如下：  
[https://github.com/kleinlee/DH_live/blob/main/README.md]

