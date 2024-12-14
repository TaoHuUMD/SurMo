from ..base_options.train_options import TrainOptions
from .project_option import import_project_opt


class ProjectOptions(TrainOptions):
    def initialize(self):
        TrainOptions.initialize(self)
        import_project_opt(self.parser)