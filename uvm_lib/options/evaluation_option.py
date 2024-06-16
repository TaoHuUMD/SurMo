from uvm_lib.engine.lib.options.evaluate_options import EvaluateOptions
from .project_option import import_project_opt


class ProjectOptions(EvaluateOptions):
    def initialize(self):
        EvaluateOptions.initialize(self)
        import_project_opt(self.parser)