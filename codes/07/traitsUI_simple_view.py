# -*- coding: utf-8 -*-
from enthought.traits.api import HasTraits, Str, Int
from enthought.traits.ui.api import View, Item 

class Employee(HasTraits):
    name = Str
    department = Str
    salary = Int
    bonus = Int
    
    view = View( 
        Item('department', label="部门", tooltip="在哪个部门干活"), 
        Item('name', label="姓名"),
        Item('salary', label="工资"),
        Item('bonus', label="奖金"),
        title = "员工资料", width=250, height=150, resizable=True 
    )
    
if __name__ == "__main__":
    p = Employee()
    p.configure_traits()