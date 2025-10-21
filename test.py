"""
🔴 安全漏洞测试代码
⚠️ 警告：此代码仅用于测试 CodeQL 检测能力，请勿用于生产环境！

预期 CodeQL 应该检测到以下问题：
1. SQL 注入 (CWE-89)
2. XSS 跨站脚本 (CWE-79)
3. 路径遍历 (CWE-22)
4. 命令注入 (CWE-78)
5. 不安全的反序列化 (CWE-502)
6. 硬编码密钥 (CWE-798)
7. 弱加密算法 (CWE-327)
8. 不安全的随机数 (CWE-330)
9. 异常处理过于宽泛
10. 未使用的导入
11. 资源泄露
12. 明文存储敏感信息
"""

import sqlite3
import pickle
import os
import subprocess
import random
import hashlib  # ❌ 未使用的导入 - CodeQL 应该检测到
from flask import Flask, request, render_template_string

app = Flask(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. SQL 注入漏洞 (CWE-89) 🔴
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def vulnerable_user_login(username, password):
    """
    ❌ 漏洞：使用字符串拼接构建 SQL 查询
    攻击示例：username = "admin' --"
    """
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # 🔴 SQL 注入：直接拼接用户输入
    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
    cursor.execute(query)
    
    result = cursor.fetchone()
    conn.close()
    return result


def vulnerable_search_user(user_id):
    """
    ❌ 漏洞：使用 % 格式化构建 SQL
    """
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # 🔴 SQL 注入：% 格式化
    query = "SELECT * FROM users WHERE id = %s" % user_id
    cursor.execute(query)
    
    return cursor.fetchall()


def vulnerable_dynamic_query(table_name, column_name, value):
    """
    ❌ 漏洞：动态表名和列名
    """
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # 🔴 SQL 注入：动态表名和列名
    query = f"SELECT * FROM {table_name} WHERE {column_name} = '{value}'"
    cursor.execute(query)
    
    return cursor.fetchall()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. XSS 跨站脚本 (CWE-79) 🔴
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.route('/search')
def vulnerable_search():
    """
    ❌ 漏洞：直接将用户输入渲染到 HTML
    攻击示例：?q=<script>alert('XSS')</script>
    """
    search_query = request.args.get('q', '')
    
    # 🔴 XSS：未转义的用户输入
    html = f"<h1>搜索结果：{search_query}</h1>"
    return html


@app.route('/profile')
def vulnerable_profile():
    """
    ❌ 漏洞：render_template_string 直接渲染用户输入
    """
    username = request.args.get('name', 'Guest')
    
    # 🔴 XSS：模板注入
    template = f"<h1>欢迎 {username}</h1>"
    return render_template_string(template)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 路径遍历 (CWE-22) 🔴
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def vulnerable_read_file(filename):
    """
    ❌ 漏洞：未验证文件路径
    攻击示例：filename = "../../etc/passwd"
    """
    # 🔴 路径遍历：直接拼接用户输入的路径
    filepath = f"/var/www/uploads/{filename}"
    
    with open(filepath, 'r') as f:
        return f.read()


def vulnerable_file_download(user_file):
    """
    ❌ 漏洞：未规范化路径
    """
    # 🔴 路径遍历：可以访问任意文件
    base_dir = "/home/user/documents/"
    full_path = base_dir + user_file
    
    return open(full_path, 'rb').read()


@app.route('/download')
def vulnerable_download():
    """
    ❌ 漏洞：Web 路径遍历
    """
    filename = request.args.get('file')
    
    # 🔴 路径遍历：未验证文件名
    return open(f"./files/{filename}", 'r').read()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 命令注入 (CWE-78) 🔴
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def vulnerable_ping(hostname):
    """
    ❌ 漏洞：未验证就执行系统命令
    攻击示例：hostname = "google.com; rm -rf /"
    """
    # 🔴 命令注入：使用 shell=True 执行用户输入
    command = f"ping -c 4 {hostname}"
    result = subprocess.run(command, shell=True, capture_output=True)
    
    return result.stdout.decode()


def vulnerable_backup(directory):
    """
    ❌ 漏洞：使用 os.system 执行命令
    """
    # 🔴 命令注入：os.system 执行拼接的命令
    os.system(f"tar -czf backup.tar.gz {directory}")


def vulnerable_convert_image(input_file, output_file):
    """
    ❌ 漏洞：使用 popen 执行命令
    """
    # 🔴 命令注入：popen 执行用户输入
    cmd = f"convert {input_file} {output_file}"
    os.popen(cmd)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. 不安全的反序列化 (CWE-502) 🔴
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def vulnerable_load_data(serialized_data):
    """
    ❌ 漏洞：反序列化不可信数据
    攻击：可以执行任意代码
    """
    # 🔴 不安全的反序列化：pickle.loads 可以执行任意代码
    data = pickle.loads(serialized_data)
    return data


def vulnerable_load_from_file(filename):
    """
    ❌ 漏洞：从文件加载 pickle 数据
    """
    with open(filename, 'rb') as f:
        # 🔴 不安全的反序列化
        return pickle.load(f)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. 硬编码密钥 (CWE-798) 🔴
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 🔴 硬编码的 API 密钥
API_KEY = "sk-1234567890abcdef1234567890abcdef"
DATABASE_PASSWORD = "MySecretPassword123!"
AWS_SECRET_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"

def vulnerable_api_call(endpoint):
    """
    ❌ 漏洞：硬编码的凭证
    """
    # 🔴 硬编码密钥
    headers = {
        "Authorization": f"Bearer sk-proj-abcd1234567890",
        "API-Key": "AIzaSyD1234567890abcdefghijklmnop"
    }
    # ... 发送请求


def vulnerable_database_connect():
    """
    ❌ 漏洞：硬编码数据库密码
    """
    # 🔴 硬编码凭证
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("ATTACH DATABASE 'secure.db' AS secure KEY 'hardcoded_password_123'")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. 弱加密算法 (CWE-327) 🔴
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import hashlib
from Crypto.Cipher import DES

def vulnerable_hash_password(password):
    """
    ❌ 漏洞：使用 MD5 哈希密码
    """
    # 🔴 弱加密：MD5 已被破解
    return hashlib.md5(password.encode()).hexdigest()


def vulnerable_encrypt_data(data, key):
    """
    ❌ 漏洞：使用 DES 加密
    """
    # 🔴 弱加密：DES 密钥太短，容易破解
    cipher = DES.new(key, DES.MODE_ECB)
    return cipher.encrypt(data)


def vulnerable_sha1_hash(data):
    """
    ❌ 漏洞：使用 SHA1
    """
    # 🔴 弱加密：SHA1 已不安全
    return hashlib.sha1(data.encode()).hexdigest()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. 不安全的随机数 (CWE-330) 🔴
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def vulnerable_generate_token():
    """
    ❌ 漏洞：使用 random 生成安全令牌
    """
    # 🔴 不安全的随机数：random 是伪随机，可预测
    token = ''.join([str(random.randint(0, 9)) for _ in range(32)])
    return token


def vulnerable_session_id():
    """
    ❌ 漏洞：使用 random.random() 生成会话ID
    """
    # 🔴 不安全的随机数
    return str(random.random())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. 异常处理问题 🔴
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def vulnerable_exception_handling(data):
    """
    ❌ 漏洞：捕获所有异常，包括系统退出信号
    """
    try:
        process_data(data)
    except:  # 🔴 过于宽泛的异常处理
        pass  # 默默吞掉所有错误


def vulnerable_bare_except():
    """
    ❌ 漏洞：裸 except 会捕获 KeyboardInterrupt
    """
    try:
        risky_operation()
    except:  # 🔴 会捕获 Ctrl+C
        print("发生错误")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. 资源泄露 🔴
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def vulnerable_file_handling(filename):
    """
    ❌ 漏洞：文件未关闭
    """
    # 🔴 资源泄露：文件未使用 with 语句
    f = open(filename, 'r')
    data = f.read()
    # 忘记 f.close()
    return data


def vulnerable_database_connection():
    """
    ❌ 漏洞：数据库连接未关闭
    """
    # 🔴 资源泄露：连接未关闭
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    return cursor.fetchall()
    # 忘记 conn.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 11. 明文存储敏感信息 (CWE-312) 🔴
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def vulnerable_save_password(username, password):
    """
    ❌ 漏洞：明文存储密码
    """
    # 🔴 明文存储：密码应该加密或哈希
    with open('passwords.txt', 'a') as f:
        f.write(f"{username}:{password}\n")


def vulnerable_log_credentials(api_key):
    """
    ❌ 漏洞：在日志中记录敏感信息
    """
    # 🔴 明文日志：API密钥不应该记录
    print(f"使用 API Key: {api_key}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 12. SSRF 服务器端请求伪造 🔴
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import requests

def vulnerable_fetch_url(url):
    """
    ❌ 漏洞：未验证 URL，可能访问内部服务
    攻击示例：url = "http://localhost:6379/config/set"
    """
    # 🔴 SSRF：直接请求用户提供的 URL
    response = requests.get(url)
    return response.text


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 辅助函数（假设存在）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def process_data(data):
    """占位函数"""
    pass

def risky_operation():
    """占位函数"""
    pass


if __name__ == "__main__":
    print("⚠️  这是一个漏洞测试文件，不要在生产环境运行！")
    print("📋 包含以下漏洞类型：")
    print("   1. SQL 注入 (3个)")
    print("   2. XSS 跨站脚本 (2个)")
    print("   3. 路径遍历 (3个)")
    print("   4. 命令注入 (3个)")
    print("   5. 不安全的反序列化 (2个)")
    print("   6. 硬编码密钥 (3个)")
    print("   7. 弱加密算法 (3个)")
    print("   8. 不安全的随机数 (2个)")
    print("   9. 异常处理问题 (2个)")
    print("   10. 资源泄露 (2个)")
    print("   11. 明文存储敏感信息 (2个)")
    print("   12. SSRF (1个)")
    print("   13. 未使用的导入 (1个)")
    print("\n   总计: 29+ 个安全问题")
    print("\n🔍 提交此文件的 PR，观察 CodeQL 能检测到多少问题！")