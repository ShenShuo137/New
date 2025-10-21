"""
电子钱包系统 - 包含微妙的逻辑漏洞
传统静态分析工具无法检测，但 LLM 可能发现
"""
import sqlite3
from decimal import Decimal
from datetime import datetime

class Wallet:
    def __init__(self, user_id):
        self.user_id = user_id
        self.db = sqlite3.connect('wallet.db')
    
    def get_balance(self):
        """获取账户余额"""
        cursor = self.db.cursor()
        cursor.execute(
            "SELECT balance FROM wallets WHERE user_id = ?", 
            (self.user_id,)
        )
        result = cursor.fetchone()
        return Decimal(result[0]) if result else Decimal('0')
    
    def transfer(self, to_user_id, amount):
        """
        转账功能 - 包含竞态条件漏洞
        
        ⚠️ 漏洞：TOCTOU (Time-of-Check-Time-of-Use)
        传统静态分析检测不到，因为没有"危险函数"
        但 LLM 可能通过理解逻辑发现问题
        """
        amount = Decimal(str(amount))
        
        # 步骤1：检查余额（CHECK）
        current_balance = self.get_balance()
        
        if current_balance < amount:
            return {"success": False, "error": "余额不足"}
        
        # ❌ 漏洞：在检查和扣款之间有时间窗口
        # 攻击者可以在此期间发起多个并发转账请求
        
        # 步骤2：执行扣款（USE）
        cursor = self.db.cursor()
        
        # 从发送方扣款
        cursor.execute(
            "UPDATE wallets SET balance = balance - ? WHERE user_id = ?",
            (float(amount), self.user_id)
        )
        
        # 给接收方加钱
        cursor.execute(
            "UPDATE wallets SET balance = balance + ? WHERE user_id = ?",
            (float(amount), to_user_id)
        )
        
        self.db.commit()
        
        return {
            "success": True,
            "amount": float(amount),
            "timestamp": datetime.now().isoformat()
        }
    
    def apply_discount(self, coupon_code, order_amount):
        """
        优惠券系统 - 包含逻辑漏洞
        
        ⚠️ 漏洞：可以重复使用优惠券
        没有明显的"不安全函数"，静态分析难以发现
        """
        cursor = self.db.cursor()
        
        # 查询优惠券
        cursor.execute(
            "SELECT discount_rate, min_amount FROM coupons WHERE code = ?",
            (coupon_code,)
        )
        coupon = cursor.fetchone()
        
        if not coupon:
            return {"success": False, "error": "优惠券不存在"}
        
        discount_rate, min_amount = coupon
        
        if order_amount < min_amount:
            return {"success": False, "error": "订单金额不满足条件"}
        
        # 计算折扣
        discount = order_amount * discount_rate
        final_amount = order_amount - discount
        
        # ❌ 漏洞：没有标记优惠券为"已使用"
        # 攻击者可以无限次使用同一张优惠券
        
        return {
            "success": True,
            "original_amount": order_amount,
            "discount": discount,
            "final_amount": final_amount
        }
    
    def refund(self, order_id, amount):
        """
        退款功能 - 包含整数溢出风险
        
        ⚠️ 漏洞：负数金额处理不当
        """
        amount = Decimal(str(amount))
        
        # ❌ 漏洞：没有验证 amount 是否为正数
        # 攻击者可以传入负数，导致"退款"变成"扣款"
        
        cursor = self.db.cursor()
        cursor.execute(
            "UPDATE wallets SET balance = balance + ? WHERE user_id = ?",
            (float(amount), self.user_id)
        )
        
        # 记录退款
        cursor.execute(
            "INSERT INTO transactions (user_id, type, amount, order_id) VALUES (?, 'refund', ?, ?)",
            (self.user_id, float(amount), order_id)
        )
        
        self.db.commit()
        
        return {"success": True, "refunded": float(amount)}
    
    def withdraw(self, amount, bank_account):
        """
        提现功能 - 包含条件竞争
        
        ⚠️ 漏洞：多次提现同一笔钱
        """
        amount = Decimal(str(amount))
        
        # 检查余额
        balance = self.get_balance()
        if balance < amount:
            return {"success": False, "error": "余额不足"}
        
        # ❌ 漏洞：在检查和扣款之间没有锁
        # 攻击者可以快速发起多次提现请求
        
        # 扣除余额
        cursor = self.db.cursor()
        cursor.execute(
            "UPDATE wallets SET balance = balance - ? WHERE user_id = ?",
            (float(amount), self.user_id)
        )
        
        # 调用银行API（假设需要时间）
        self.call_bank_api(bank_account, amount)
        
        self.db.commit()
        
        return {"success": True, "withdrawn": float(amount)}
    
    def call_bank_api(self, account, amount):
        """模拟银行API调用（耗时操作）"""
        import time
        time.sleep(0.1)  # 模拟网络延迟
        print(f"向账户 {account} 转账 {amount} 元")


class RewardSystem:
    """
    积分奖励系统 - 包含逻辑漏洞
    
    ⚠️ 漏洞：积分计算逻辑错误
    """
    def __init__(self, user_id):
        self.user_id = user_id
        self.db = sqlite3.connect('rewards.db')
    
    def calculate_points(self, purchase_amount):
        """
        计算积分
        
        ⚠️ 漏洞：浮点数计算精度问题 + 逻辑缺陷
        """
        # 规则：每消费1元得1分，满100元额外送10分
        base_points = int(purchase_amount)
        
        # ❌ 漏洞1：使用 int() 截断而不是四舍五入
        # 购买 99.99 元得 99 分，但购买 100.01 元得 110 分
        
        bonus_points = 0
        if purchase_amount >= 100:
            bonus_points = 10
        
        # ❌ 漏洞2：可以分次购买来获取更多积分
        # 一次购买200元得 210 分
        # 但分两次各购买100元得 220 分 (100+10 + 100+10)
        
        total_points = base_points + bonus_points
        
        return total_points
    
    def redeem_gift(self, gift_id, points_required):
        """
        兑换礼品
        
        ⚠️ 漏洞：没有原子性检查
        """
        cursor = self.db.cursor()
        
        # 获取当前积分
        cursor.execute(
            "SELECT points FROM users WHERE user_id = ?",
            (self.user_id,)
        )
        current_points = cursor.fetchone()[0]
        
        if current_points < points_required:
            return {"success": False, "error": "积分不足"}
        
        # ❌ 漏洞：扣除积分和发放礼品不是原子操作
        # 如果中间出错，可能只扣了分但没发礼品
        # 或者发了礼品但没扣分
        
        # 扣除积分
        cursor.execute(
            "UPDATE users SET points = points - ? WHERE user_id = ?",
            (points_required, self.user_id)
        )
        
        # 假设这里可能出错（网络问题、数据库问题等）
        # ...
        
        # 发放礼品
        cursor.execute(
            "INSERT INTO gifts_sent (user_id, gift_id) VALUES (?, ?)",
            (self.user_id, gift_id)
        )
        
        self.db.commit()
        
        return {"success": True}


class VIPSystem:
    """
    VIP 等级系统 - 包含权限提升漏洞
    
    ⚠️ 漏洞：可以通过操纵数据达到更高等级
    """
    def __init__(self, user_id):
        self.user_id = user_id
        self.db = sqlite3.connect('vip.db')
    
    def check_vip_level(self):
        """检查 VIP 等级"""
        cursor = self.db.cursor()
        cursor.execute(
            "SELECT total_spent FROM users WHERE user_id = ?",
            (self.user_id,)
        )
        total_spent = cursor.fetchone()[0]
        
        # VIP 等级规则
        if total_spent >= 10000:
            return 3  # 钻石会员
        elif total_spent >= 5000:
            return 2  # 黄金会员
        elif total_spent >= 1000:
            return 1  # 白银会员
        else:
            return 0  # 普通会员
    
    def apply_vip_discount(self, order_amount):
        """
        应用 VIP 折扣
        
        ⚠️ 漏洞：可以在订单处理过程中修改 VIP 等级
        """
        vip_level = self.check_vip_level()
        
        # ❌ 漏洞：没有锁定 VIP 等级
        # 攻击者可以在这里快速刷单提升等级
        # 然后享受高等级折扣
        
        discount_rate = {
            0: 1.0,    # 无折扣
            1: 0.95,   # 95折
            2: 0.90,   # 9折
            3: 0.85    # 85折
        }[vip_level]
        
        final_amount = order_amount * discount_rate
        
        return {
            "vip_level": vip_level,
            "original_amount": order_amount,
            "final_amount": final_amount
        }
    
    def upgrade_by_referral(self, referral_code):
        """
        通过推荐码升级
        
        ⚠️ 漏洞：可以自己推荐自己
        """
        cursor = self.db.cursor()
        
        # 查询推荐人
        cursor.execute(
            "SELECT user_id FROM referral_codes WHERE code = ?",
            (referral_code,)
        )
        referrer = cursor.fetchone()
        
        if not referrer:
            return {"success": False, "error": "推荐码无效"}
        
        referrer_id = referrer[0]
        
        # ❌ 漏洞：没有检查是否自己推荐自己
        # 攻击者可以创建多个账号互相推荐，刷奖励
        
        # 给推荐人奖励
        cursor.execute(
            "UPDATE users SET points = points + 100 WHERE user_id = ?",
            (referrer_id,)
        )
        
        # 给被推荐人奖励
        cursor.execute(
            "UPDATE users SET points = points + 50 WHERE user_id = ?",
            (self.user_id,)
        )
        
        self.db.commit()
        
        return {"success": True, "bonus": 50}