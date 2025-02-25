#!/bin/bash

# 创建安装目录
mkdir -p $HOME/java

# 下载JDK 17 (使用Amazon Corretto发行版，它是免费的)
wget -O $HOME/java/jdk17.tar.gz https://corretto.aws/downloads/latest/amazon-corretto-17-x64-linux-jdk.tar.gz

# 解压JDK
cd $HOME/java
tar -xzf jdk17.tar.gz
rm jdk17.tar.gz

# 获取解压后的目录名
JDK_DIR=$(ls -d */ | grep -i corretto)

# 添加环境变量配置到.bashrc和.zshrc
for rcfile in "$HOME/.bashrc" "$HOME/.zshrc"; do
    if [ -f "$rcfile" ]; then
        # 移除旧的Java环境变量配置
        sed -i '/# Java Environment Variables/d' "$rcfile"
        sed -i '/JAVA_HOME/d' "$rcfile"

        # 添加新的环境变量配置
        echo "\n# Java Environment Variables" >> "$rcfile"
        echo "export JAVA_HOME=\"$HOME/java/$JDK_DIR\"" >> "$rcfile"
        echo 'export PATH="$JAVA_HOME/bin:$PATH"' >> "$rcfile"
    fi
done

# 立即设置环境变量
export JAVA_HOME="$HOME/java/$JDK_DIR"
export PATH="$JAVA_HOME/bin:$PATH"

# 验证安装
java -version