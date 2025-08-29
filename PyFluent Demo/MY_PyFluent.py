# -*- coding: utf-8 -*-
import ansys.fluent.core as pyfluent
import csv
import os

# 创建列表读取数据
data_all = []
with open ("D:\\MY_Project\\MY_pyfluent\\MY_Nozzle\\MY_Data_Generator\\MY_Input_Data\\MY_Data.csv",'r',encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data_all.append(row)

# 读取模板mesh文件
filename = "D:\\MY_Project\\MY_pyfluent\\MY_Nozzle\\MY_Data_Generator\\case\\MY_Nozzle.cas"
solver = pyfluent.launch_fluent(version = "2d", processor_count= 4, show_gui= False, mode="solver")
solver.file.read(file_type = "case", file_name = filename)

# 根据 data 数据里的 case 设置边界条件和迭代次数
for data in data_all:
    print(data)
    cases = int(data['cases'])
    Pressure = int(data['Pressure'])
    Temperature = int(data['Temperature'])

    # 动态生成唯一文件名
    export_path = "D:\\MY_Project\\MY_pyfluent\\MY_Nozzle\\MY_Data_Generator\\MY_Output_Data"
    file_name = f"{Pressure:.0f},{Temperature:.0f}.txt"
    file_path = os.path.join(export_path, file_name)

    print(f"正在运行: {data['cases']}, 压力为: {Pressure},温度为：: {Temperature}")

    # 重新加载 .cas 文件
    solver.file.read(file_type="case", file_name=filename)

    # 改一下小数据
    solver.tui.define.operating_conditions.operating_pressure('0')

    try:

        # 设置边界条件
        solver.tui.define.boundary_conditions.pressure_inlet(
            "inlet", "yes", "no", Pressure, 'no', Pressure, 'no',Temperature, "no", 'yes',
            "no", "no", "no", "yes", "5", "0.38"
        )
        solver.tui.define.boundary_conditions.pressure_outlet(
            "outlet", "yes", "no", "101325", "no", "300", "no", "yes", "no", "no", "yes", "5", "10", "yes", "no", "no"
        )

        # 初始化流场
        solver.tui.solve.initialize.compute_defaults.pressure_inlet("inlet")
        solver.tui.solve.initialize.initialize_flow()
        print('初始化完成')

        # 求解器运行
        solver.tui.solve.iterate(1000)

        # 导出仿真结果
        solver.tui.file.export.ascii(file_path, "body", "()", "no", "pressure", "mach-number", "temperature", "()", "no")

        print(f"Exported results to: {file_path}")

    except Exception as e:
        print(f"Error in case {data['cases']}: {e}")

    solver.tui.file.write_case_data(str(Pressure) + str(Temperature) + '.cas')
    print(str(cases) + " 运行结束")

# 退出 Fluent 求解器
solver.exit()