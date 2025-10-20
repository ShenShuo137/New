"""
用户认证模块 - 测试PR-Agent审查能力
包含一些故意的安全问题和代码质量问题
"""
import sqlite3
import hashlib

class UserAuth:
    def __init__(self):
        self.db_path = 'users.db'
    
    def login(self, username, password):
        """用户登录 - 存在SQL注入漏洞"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # ❌ 严重安全问题：SQL注入
        query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
        cursor.execute(query)
        
        user = cursor.fetchone()
        conn.close()
        
        if user:
            print(f"用户 {username} 登录成功")  # ❌ 不应该打印敏感信息
            return True
        return False
    
    def register(self, username, password, email):
        """注册新用户 - 多个安全问题"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # ❌ 密码明文存储
        # ❌ SQL注入风险
        query = f"INSERT INTO users (username, password, email) VALUES ('{username}', '{password}', '{email}')"
        
        try:
            cursor.execute(query)
            conn.commit()
        except:
            pass  # ❌ 空的异常处理
        finally:
            conn.close()
        
        return True
    
    def change_password(self, user_id, old_password, new_password):
        """修改密码 - 缺少验证"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # ❌ 没有验证旧密码是否正确
        # ❌ SQL注入
        # ❌ 新密码明文存储
        query = f"UPDATE users SET password = '{new_password}' WHERE id = {user_id}"
        cursor.execute(query)
        conn.commit()
        conn.close()
    
    def get_user_by_email(self, email):
        """通过邮箱查询用户"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # ❌ SQL注入
        result = cursor.execute(f"SELECT * FROM users WHERE email = '{email}'")
        user = result.fetchone()
        
        # ❌ 忘记关闭连接
        return user

# ❌ 没有 if __name__ == "__main__" 保护
auth = UserAuth()
auth.register("admin", "123456", "admin@example.com")