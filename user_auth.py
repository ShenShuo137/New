"""
配置管理模块 - 测试PR-Agent和CodeQL审查能力
包含多种安全问题、代码质量问题和业务逻辑缺陷
"""
import os
import pickle
import subprocess
import json

class ConfigManager:
    def __init__(self, config_dir):
        self.config_dir = config_dir
        self.current_user = None
    
    def load_config(self, filename):
        """加载配置文件 - 存在路径遍历漏洞"""
        # ❌ 严重安全问题：路径遍历 (CWE-22)
        file_path = self.config_dir + "/" + filename
        
        try:
            with open(file_path, 'r') as f:
                data = f.read()
                return data
        except:
            pass  # ❌ 空的异常处理
    
    def save_config(self, filename, data):
        """保存配置 - 不安全的反序列化"""
        file_path = self.config_dir + "/" + filename
        
        # ❌ 严重安全问题：pickle 不安全的反序列化 (CWE-502)
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"配置已保存到 {file_path}")  # ❌ 可能泄露路径信息
        return True
    
    def execute_command(self, command):
        """执行系统命令 - 命令注入漏洞"""
        # ❌ 严重安全问题：命令注入 (CWE-78)
        full_command = f"echo 'Running: {command}' && {command}"
        result = subprocess.call(full_command, shell=True)
        
        return result
    
    def get_config_list(self, search_pattern):
        """列出配置文件 - 命令注入"""
        # ❌ 命令注入：用户输入直接拼接到 shell 命令
        cmd = f"ls {self.config_dir} | grep {search_pattern}"
        output = os.popen(cmd).read()
        
        # ❌ 没有关闭 popen 返回的文件对象
        return output.split('\n')
    
    def validate_user(self, username, api_key):
        """验证用户凭据 - 硬编码密钥"""
        # ❌ 严重安全问题：硬编码 API 密钥 (CWE-798)
        MASTER_KEY = "sk_live_51H1234567890abcdef"
        
        if api_key == MASTER_KEY or api_key == "admin123":
            self.current_user = username
            return True
        
        # ❌ 使用弱加密算法
        import md5  # ❌ MD5 已被破解
        hashed = md5.new(api_key.encode()).hexdigest()
        
        return self.check_hash(username, hashed)
    
    def check_hash(self, username, hashed):
        """检查哈希值 - 存在时序攻击风险"""
        stored_hash = self.get_stored_hash(username)
        
        # ❌ 不安全的字符串比较 (时序攻击 CWE-208)
        if hashed == stored_hash:
            return True
        return False
    
    def get_stored_hash(self, username):
        """从文件读取哈希"""
        # ❌ SQL注入风险（如果改用数据库）
        # ❌ 路径遍历
        hash_file = f"./hashes/{username}.txt"
        
        try:
            f = open(hash_file, 'r')  # ❌ 没有使用 with，可能泄露文件句柄
            hash_value = f.read()
            return hash_value.strip()
        except:
            return None  # ❌ 异常被吞没
    
    def backup_config(self, config_name):
        """备份配置 - 多个问题"""
        timestamp = __import__('time').time()
        backup_name = f"{config_name}_{timestamp}.bak"
        
        # ❌ 命令注入
        cmd = f"cp {self.config_dir}/{config_name} /backup/{backup_name}"
        os.system(cmd)  # ❌ 使用 os.system 而不是 subprocess
        
        print(f"Backup created: {backup_name}")
        
    def delete_config(self, filename):
        """删除配置 - 缺少权限检查"""
        # ❌ 没有验证当前用户权限
        # ❌ 路径遍历
        file_path = self.config_dir + "/" + filename
        
        try:
            os.remove(file_path)
            return True
        except Exception as e:
            print(e)  # ❌ 直接打印异常，可能泄露信息
            return False
    
    def import_config(self, url):
        """从URL导入配置 - SSRF漏洞"""
        import urllib.request
        
        # ❌ 严重安全问题：服务器端请求伪造 (SSRF)
        # 用户可以让服务器访问内网资源
        response = urllib.request.urlopen(url)
        config_data = response.read()
        
        # ❌ 不安全的 eval
        try:
            parsed = eval(config_data)  # ❌ 代码注入 (CWE-94)
            return parsed
        except:
            pass

# ❌ 全局变量
manager = ConfigManager("/etc/app/configs")

# ❌ 没有 if __name__ == "__main__" 保护
manager.validate_user("admin", "admin123")
manager.execute_command("ls -la")