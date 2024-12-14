from ..base_options.test_options import TestOptions
from .project_option import import_project_opt
import os

class ProjectOptions(TestOptions):
    def initialize(self):
        TestOptions.initialize(self)
        import_project_opt(self.parser)

    def parse_(self):
        opt = self.parse()

        opt.results_dir = os.path.join(opt.results_dir, opt.name)
