define(["require", "jquery", "base/js/namespace"], function (require, $, IPython) {
    var load_css = function () {
        var link = document.createElement("link");
        link.type = "text/css";
        link.rel = "stylesheet";
        link.href = require.toUrl("./scipybook2.css");
        document.getElementsByTagName("head")[0].appendChild(link);
    };    
    
    var load_ipython_extension = function () {
        load_css();
        IPython.load_extensions('scipybook2/info_blocks');
        /*IPython.load_extensions('scipybook2/key_macros');*/
        /*IPython.load_extensions('scipybook2/section_copy');*/
        /*IPython.load_extensions('scipybook2/simple_ui');*/

        IPython.CodeCell.config_defaults.highlight_modes['magic_text/x-csrc'] = {'reg':[/^%%language c/, /^%%include c /]};
        IPython.CodeCell.config_defaults.highlight_modes['magic_text/x-cython'].reg.push(/^%%include cython/);
        console.log("scipybook2 js loaded");
    };

  return {
    load_ipython_extension : load_ipython_extension,
  };
});
