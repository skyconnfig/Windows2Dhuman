@echo off
echo ========================================
echo    2D数字人项目 - 虚拟环境激活脚本
echo ========================================
echo.

REM 检查虚拟环境是否存在
if not exist "venv\Scripts\activate.bat" (
    echo [错误] 虚拟环境不存在！
    echo 请先运行以下命令创建虚拟环境：
    echo python -m venv venv
    pause
    exit /b 1
)

REM 激活虚拟环境
echo [信息] 正在激活Python虚拟环境...
call venv\Scripts\activate.bat

REM 显示环境信息
echo.
echo [成功] 虚拟环境已激活！
echo.
echo 当前Python版本：
python --version
echo.
echo 虚拟环境路径：
echo %VIRTUAL_ENV%
echo.
echo ========================================
echo 可用命令：
echo   python test_env.py        - 测试环境配置
echo   python app_simple.py      - 运行简化版应用
echo   python app.py             - 运行完整应用
echo   deactivate                - 退出虚拟环境
echo ========================================
echo.

REM 保持命令行窗口打开
cmd /k