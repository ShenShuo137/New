/**
 * 配置管理类 - 测试PR-Agent和CodeQL审查能力
 * 包含多种安全问题、代码质量问题和业务逻辑缺陷
 * 
 * @author Test
 * @version 1.0
 */

 package com.example.config;

 import java.io.*;
 import java.net.URL;
 import java.nio.file.*;
 import java.security.MessageDigest;
 import java.sql.*;
 import java.util.*;
 
 public class ConfigManager {
     private String configDir;
     private String currentUser;
     
     // ❌ 严重安全问题：硬编码密钥 (CWE-798)
     private static final String MASTER_KEY = "sk_live_51H1234567890abcdef";
     private static final String ADMIN_PASSWORD = "admin123";
     
     // ❌ 使用不安全的随机数生成器
     private Random random = new Random();  // 应使用 SecureRandom
     
     public ConfigManager(String configDir) {
         this.configDir = configDir;
         this.currentUser = null;
     }
     
     /**
      * 加载配置文件 - 存在路径遍历漏洞
      * ❌ 严重安全问题：路径遍历 (CWE-22)
      */
     public String loadConfig(String filename) {
         // ❌ 未验证文件名，可能包含 "../"
         String filePath = configDir + File.separator + filename;
         
         StringBuilder content = new StringBuilder();
         
         try {
             // ❌ 资源泄露：没有使用 try-with-resources
             BufferedReader reader = new BufferedReader(new FileReader(filePath));
             String line;
             
             while ((line = reader.readLine()) != null) {
                 content.append(line).append("\n");
             }
             
             // ❌ 忘记关闭 reader
             
         } catch (Exception e) {
             // ❌ 空的异常处理
             e.printStackTrace();  // ❌ 可能泄露堆栈信息
         }
         
         return content.toString();
     }
     
     /**
      * 保存配置 - 不安全的序列化
      * ❌ 严重安全问题：不安全的反序列化 (CWE-502)
      */
     public boolean saveConfig(String filename, Object data) {
         String filePath = configDir + File.separator + filename;
         
         try {
             // ❌ Java 原生序列化不安全
             FileOutputStream fileOut = new FileOutputStream(filePath);
             ObjectOutputStream out = new ObjectOutputStream(fileOut);
             
             out.writeObject(data);  // ❌ 反序列化漏洞
             
             out.close();
             fileOut.close();
             
             // ❌ 可能泄露路径信息
             System.out.println("配置已保存到 " + filePath);
             
             return true;
         } catch (IOException e) {
             e.printStackTrace();
             return false;
         }
     }
     
     /**
      * 执行系统命令 - 命令注入漏洞
      * ❌ 严重安全问题：命令注入 (CWE-78)
      */
     public int executeCommand(String command) throws IOException {
         // ❌ 命令注入：用户输入直接传递给 Runtime.exec
         String[] cmd;
         
         if (System.getProperty("os.name").toLowerCase().contains("win")) {
             cmd = new String[]{"cmd.exe", "/c", command};
         } else {
             // ❌ 使用 shell 执行，可能被注入
             cmd = new String[]{"/bin/sh", "-c", command};
         }
         
         Process process = Runtime.getRuntime().exec(cmd);
         
         try {
             return process.waitFor();
         } catch (InterruptedException e) {
             Thread.currentThread().interrupt();
             return -1;
         }
     }
     
     /**
      * SQL 查询用户 - SQL注入漏洞
      * ❌ 严重安全问题：SQL注入 (CWE-89)
      */
     public User getUserByUsername(Connection conn, String username) {
         try {
             // ❌ SQL注入：字符串拼接
             String query = "SELECT * FROM users WHERE username = '" + username + "'";
             
             Statement stmt = conn.createStatement();  // ❌ 应使用 PreparedStatement
             ResultSet rs = stmt.executeQuery(query);
             
             if (rs.next()) {
                 User user = new User();
                 user.setUsername(rs.getString("username"));
                 user.setEmail(rs.getString("email"));
                 
                 // ❌ 资源泄露：ResultSet 和 Statement 未关闭
                 return user;
             }
             
         } catch (SQLException e) {
             e.printStackTrace();  // ❌ 可能泄露数据库结构信息
         }
         
         return null;
     }
     
     /**
      * 验证用户凭据 - 多个安全问题
      */
     public boolean validateUser(String username, String apiKey) {
         // ❌ 使用硬编码的密钥
         if (apiKey.equals(MASTER_KEY) || apiKey.equals(ADMIN_PASSWORD)) {
             this.currentUser = username;
             return true;
         }
         
         // ❌ 使用已弃用的 MD5 算法
         try {
             MessageDigest md = MessageDigest.getInstance("MD5");
             byte[] hashBytes = md.digest(apiKey.getBytes());
             
             String hashed = bytesToHex(hashBytes);
             
             return checkHash(username, hashed);
             
         } catch (Exception e) {
             return false;
         }
     }
     
     /**
      * 检查哈希值 - 时序攻击风险
      * ❌ 不安全的字符串比较 (CWE-208)
      */
     public boolean checkHash(String username, String hashed) {
         String storedHash = getStoredHash(username);
         
         // ❌ 使用 equals 比较，可能存在时序攻击
         if (hashed.equals(storedHash)) {
             return true;
         }
         
         return false;
     }
     
     /**
      * 从文件读取哈希 - 路径遍历
      */
     public String getStoredHash(String username) {
         // ❌ 路径遍历：未验证 username
         String hashFile = "./hashes/" + username + ".txt";
         
         try {
             // ❌ 资源泄露：没有使用 try-with-resources
             BufferedReader reader = new BufferedReader(new FileReader(hashFile));
             String hash = reader.readLine();
             
             // ❌ 忘记关闭 reader
             
             return hash;
         } catch (IOException e) {
             // ❌ 异常被吞没
             return null;
         }
     }
     
     /**
      * 从URL导入配置 - SSRF漏洞
      * ❌ 严重安全问题：服务器端请求伪造 (CWE-918)
      */
     public String importConfig(String urlString) {
         try {
             // ❌ SSRF：未验证 URL，可能访问内网资源
             URL url = new URL(urlString);
             
             BufferedReader in = new BufferedReader(
                 new InputStreamReader(url.openStream())
             );
             
             StringBuilder content = new StringBuilder();
             String line;
             
             while ((line = in.readLine()) != null) {
                 content.append(line);
             }
             
             // ❌ 资源泄露：未关闭 BufferedReader
             
             return content.toString();
             
         } catch (Exception e) {
             e.printStackTrace();
             return null;
         }
     }
     
     /**
      * XML 解析 - XXE 漏洞
      * ❌ 严重安全问题：XML 外部实体注入 (CWE-611)
      */
     public void parseXMLConfig(String xmlContent) {
         try {
             // ❌ XXE：未禁用外部实体
             javax.xml.parsers.DocumentBuilderFactory factory = 
                 javax.xml.parsers.DocumentBuilderFactory.newInstance();
             
             // 应该设置这些来防止 XXE：
             // factory.setFeature("http://apache.org/xml/features/disallow-doctype-decl", true);
             
             javax.xml.parsers.DocumentBuilder builder = factory.newDocumentBuilder();
             
             ByteArrayInputStream input = new ByteArrayInputStream(xmlContent.getBytes());
             org.w3c.dom.Document doc = builder.parse(input);
             
             // 处理 XML...
             
         } catch (Exception e) {
             e.printStackTrace();
         }
     }
     
     /**
      * 删除配置 - 缺少权限检查
      * ❌ 缺少授权检查
      */
     public boolean deleteConfig(String filename) {
         // ❌ 没有验证当前用户权限
         // ❌ 路径遍历
         String filePath = configDir + File.separator + filename;
         
         File file = new File(filePath);
         
         try {
             return file.delete();
         } catch (Exception e) {
             System.err.println(e.getMessage());  // ❌ 可能泄露信息
             return false;
         }
     }
     
     /**
      * 备份配置 - 命令注入
      */
     public void backupConfig(String configName) throws IOException {
         long timestamp = System.currentTimeMillis();
         String backupName = configName + "_" + timestamp + ".bak";
         
         // ❌ 命令注入
         String cmd = "cp " + configDir + "/" + configName + " /backup/" + backupName;
         
         Runtime.getRuntime().exec(cmd);  // ❌ 使用 shell 执行
         
         System.out.println("Backup created: " + backupName);
     }
     
     /**
      * 生成临时文件 - 不安全的临时文件
      * ❌ 不安全的临时文件创建 (CWE-377)
      */
     public File createTempFile(String prefix) throws IOException {
         // ❌ 使用可预测的文件名
         String tempFileName = "/tmp/" + prefix + "_" + System.currentTimeMillis() + ".tmp";
         
         File tempFile = new File(tempFileName);
         tempFile.createNewFile();  // ❌ 没有设置适当的权限
         
         return tempFile;
     }
     
     /**
      * 竞态条件示例
      * ❌ 竞态条件 (CWE-362)
      */
     public void updateCounter(String counterFile) throws IOException {
         File file = new File(counterFile);
         
         // ❌ TOCTOU: Time-of-check to time-of-use
         if (file.exists()) {
             // 在检查和使用之间，文件可能被修改
             BufferedReader reader = new BufferedReader(new FileReader(file));
             int count = Integer.parseInt(reader.readLine());
             reader.close();
             
             count++;
             
             // 写回时文件可能已经被其他线程修改
             PrintWriter writer = new PrintWriter(new FileWriter(file));
             writer.println(count);
             writer.close();
         }
     }
     
     // ═══════════════════════════════════════════════════════════
     // 辅助方法
     // ═══════════════════════════════════════════════════════════
     
     private String bytesToHex(byte[] bytes) {
         StringBuilder result = new StringBuilder();
         for (byte b : bytes) {
             result.append(String.format("%02x", b));
         }
         return result.toString();
     }
     
     // ═══════════════════════════════════════════════════════════
     // 内部类
     // ═══════════════════════════════════════════════════════════
     
     public static class User {
         private String username;
         private String email;
         
         // ❌ 缺少访问控制修饰符
         String password;  // 应该是 private
         
         public String getUsername() { return username; }
         public void setUsername(String username) { this.username = username; }
         
         public String getEmail() { return email; }
         public void setEmail(String email) { this.email = email; }
         
         // ❌ 密码的 getter 不应该存在
         public String getPassword() { return password; }
         public void setPassword(String password) { this.password = password; }
     }
     
     // ═══════════════════════════════════════════════════════════
     // Main 方法 - 测试代码
     // ═══════════════════════════════════════════════════════════
     
     public static void main(String[] args) {
         ConfigManager manager = new ConfigManager("/etc/app/configs");
         
         // ❌ 使用硬编码凭据
         manager.validateUser("admin", "admin123");
         
         try {
             // ❌ 命令注入测试
             manager.executeCommand("ls -la");
             
             // ❌ 路径遍历测试
             String config = manager.loadConfig("../../etc/passwd");
             
         } catch (Exception e) {
             e.printStackTrace();
         }
     }
 }