# -*- coding: utf-8 -*-
"""
演示TraitsUI中的各种Group的用法
"""
from enthought.traits.api import HasTraits, Str, Int
from enthought.traits.ui.api import View, Item, Group, VGrid, VGroup, HSplit, VSplit

class SimpleEmployee(HasTraits):
    first_name = Str
    last_name = Str
    department = Str

    employee_number = Str
    salary = Int
    bonus = Int

view1 = View(
    Group(
        Item(name = 'employee_number', label='编号'),
        Item(name = 'department', label="部门", tooltip="在哪个部门干活"),
        Item(name = 'last_name', label="姓"),
        Item(name = 'first_name', label="名"),
        label = '个人信息',
        show_border = True
    ),
    Group(
        Item(name = 'salary', label="工资"),
        Item(name = 'bonus', label="奖金"),
        label = '收入',
        show_border = True
    ),
    title = "标签页方式",
    resizable = True    
)    
    
view2 = View(
    VGroup(
        VGrid(
            Item(name = 'employee_number', label='编号'),
            Item(name = 'department', label="部门", tooltip="在哪个部门干活"),
            Item(name = 'last_name', label="姓"),
            Item(name = 'first_name', label="名"),
            label = '个人信息',
            show_border = True,
            scrollable = True
        ),
        VGroup(
            Item(name = 'salary', label="工资"),
            Item(name = 'bonus', label="奖金"),
            label = '收入',
            show_border = True,
        ),
    ), 
    resizable = True,
    width = 400,
    height = 250,
    title = "垂直分组"    
)

view3 = View(
    HSplit(
        VGroup(
            Item(name = 'employee_number', label='编号'),
            Item(name = 'department', label="部门", tooltip="在哪个部门干活"),
            Item(name = 'last_name', label="姓"),
            Item(name = 'first_name', label="名"),
            show_border = True,
            scrollable = True,
        ),
        VGroup(
            Item(name = 'salary', label="工资"),
            Item(name = 'bonus', label="奖金"),
            show_border = True,
            scrollable = True,
        ),
    ), 
    resizable = True,
    width = 400,
    height = 150,
    title = "水平分组(带调节栏)"    
)

view4 = View(
    VSplit(
        VGroup(
            Item(name = 'employee_number', label='编号'),
            Item(name = 'department', label="部门", tooltip="在哪个部门干活"),
            Item(name = 'last_name', label="姓"),
            Item(name = 'first_name', label="名"),
            show_border = True,
            scrollable = True
        ),
        VGroup(
            Item(name = 'salary', label="工资"),
            Item(name = 'bonus', label="奖金"),
            show_border = True,
            scrollable = True
        ),
    ), 
    resizable = True,
    width = 200,
    height = 300,
    title = "垂直分组(带调节栏)"    
)

sam = SimpleEmployee()
sam.edit_traits(view=view1)
sam.edit_traits(view=view2)
sam.edit_traits(view=view3)
sam.configure_traits(view=view4)