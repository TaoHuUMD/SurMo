from ..engine.lib.options.test_options import TestOptions
from .project_option import import_project_opt


class ProjectOptions(TestOptions):
    def initialize(self):
        TestOptions.initialize(self)
        import_project_opt(self.parser)