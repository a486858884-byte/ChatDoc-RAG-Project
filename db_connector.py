import mysql.connector
from mysql.connector import Error

# 功能一：建立与数据库的连接
def create_db_connection(host_name, user_name, user_password, db_name):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name
        )
        print("MySQL 数据库连接成功！")
    except Error as e:
        print(f"连接失败，错误信息: '{e}'")

    return connection
# 功能二：执行“读取”类型的查询 (SELECT)
def execute_read_query(connection, query):
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except Error as e:
        print(f"查询失败，错误信息: '{e}'")
# 功能三：执行“写入”类型的查询 (INSERT, UPDATE, DELETE)
def execute_write_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit() # 极其重要！提交事务，让修改永久生效
        print("查询执行成功！")
    except Error as e:
        print(f"查询失败，错误信息: '{e}'")
# 主函数：程序的入口，我们在这里测试上面的功能
if __name__ == '__main__':
    # --- 1. 建立连接 ---
    conn = create_db_connection("localhost", "root", "1234", "chatdoc_db")
    # --- 2. 演示 INSERT (增) ---
    print("\n--- 正在插入一个新用户 'charlie' ---")
    # 我们使用 f-string 或直接拼接来构建 SQL 语句字符串
    insert_user_query = "INSERT INTO users (username) VALUES ('charlie');"
    # 调用我们编写的“写”函数来执行它
    execute_write_query(conn, insert_user_query)
    # --- 3. 演示 SELECT (查) ---
    print("\n--- 正在查询所有用户，确认 'charlie' 已添加 ---")
    select_users_query = "SELECT * FROM users;"
    # 调用我们编写的“读”函数来执行它
    users = execute_read_query(conn, select_users_query)
    # 遍历并打印查询结果
    if users:
        for user in users:
            # user 是一个元组，例如 (3, 'charlie', datetime.datetime(...))
            print(user)

    # --- 4. 演示 UPDATE (改) ---
    print("\n--- 正在将 'charlie' 的名字修改为 'charles' ---")
    update_user_query = "UPDATE users SET username = 'charles' WHERE username = 'charlie';"
    execute_write_query(conn, update_user_query)

    print("\n--- 再次查询所有用户，确认修改 ---")
    users_after_update = execute_read_query(conn, select_users_query)
    if users_after_update:
        for user in users_after_update:
            print(user)

    # --- 5. 演示 DELETE (删) ---
    print("\n--- 正在删除用户 'charles' ---")
    delete_user_query = "DELETE FROM users WHERE username = 'charles';"
    execute_write_query(conn, delete_user_query)

    print("\n--- 最后一次查询所有用户，确认删除 ---")
    users_after_delete = execute_read_query(conn, select_users_query)
    if users_after_delete:
        print("剩余用户:")
        for user in users_after_delete:
            print(user)
    else:
        # 如果表被删空了，users_after_delete 会是一个空列表
        print("表中已无用户。")

    # --- 6. 关闭连接 (非常重要的好习惯) ---
    # 检查连接对象是否存在，并且是否处于连接状态
    if conn and conn.is_connected():
        conn.close()
        print("\nMySQL 连接已关闭。")

