(function(IPython){
    var macros = {
        "1":"❶",
        "2":"❷",
        "3":"❸",
        "4":"❹",
        "5":"❺",
        "6":"❻",
        "7":"❼",
        "8":"❽",
        "9":"❾",
        "fig":'![](/files/images/.png "")',
        "next":'`ref:fig-next`',
        "prev":'`ref:fig-prev`',
        "tip":'> **TIP**\n\n> ',
        "source":'> **SOURCE**\n\n> ',
        "warning":'> **WARNING**\n\n> ',
        "question":'> **QUESTION**\n\n> ',
        "link":'> **LINK**\n\n> \n\n> ',        
     };
 
    var data = {
        help    : 'macro',
        help_index : 'aa',
        handler : function (event) {
            var cm = IPython.notebook.get_selected_cell().code_mirror;
            var cursor = cm.getCursor();
            var line = cm.getLine(cursor.line);
            var index = cursor.ch - 1;
            while(index >= 0)
            {
                if(line[index] == "$" ) break;
                index -= 1;
            }
            if(index >= 0)
            {
                var cmd = line.substring(index+1, cursor.ch);                
                var from = {line:cursor.line, ch:index};
                if (cmd in macros)
                {
                    cm.replaceRange(macros[cmd], from, cursor);
                    return false;
                }
                
                switch(cmd)
                {
                    case "1":
                        cm.replaceRange("one", from, cursor);
                        return false;
                }
                
            }
            return true;
        }    
    };
    IPython.keyboard_manager.edit_shortcuts.add_shortcut("shift-space", data, true);

}(IPython));