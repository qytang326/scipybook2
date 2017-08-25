(function(IPython){
    var compact_ui = function(){
        $("#menus .navbar-nav").append($("#move_up_down"));
        $("#menus .navbar-nav").append($("#toc_button"));
        $("#menus .navbar-nav").append($("#save_widget"));   
        $("#header .checkpoint_status").hide();
        $("#header .autosave_status").hide();   
        $("#header-container").toggle();
        $("#maintoolbar-container").toggle();
        $("#header").attr("ui_style", "compact");
        $("#save_widget").removeClass("pull-left");
        $(".navbar-nav").css("width", "830px");
    };
    
    var expand_ui = function(){
        $("#move_up_down").insertBefore($("#run_int"));
        $("#toc_button").insertBefore($("#cell_type"));
        $('#maintoolbar-container').toggle();
        $('#header-container').toggle();
        $("#save_widget").insertAfter($('#ipython_notebook'));
        $("#checkpoint_status").show();
        $("#autosave_status").show();
        $("#header .checkpoint_status").show();
        $("#header .autosave_status").show();          
        $("#header").attr("ui_style", "expand");   
        $("#save_widget").addClass("pull-left");        
        $(".navbar-nav").css("width", "");
    };
    
    $('<p class="navbar-text indicator_area">\
     <i title="Switch between compact and expanded UI" \
     class="toggle_arrow_down" id="ui_toggle_icon"></i></p>').insertAfter($("#modal_indicator"));
     
    $("#ui_toggle_icon").click(function(){
        if($("#header").attr("ui_style") == "compact")
            expand_ui();
        else
            compact_ui();
    });
     
    compact_ui();     
}(IPython));