import torch
from curator.data.utils import read_trajectory
from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
import pickle
import logging
from functools import partial

logger = logging.getLogger(__name__)
class Evaluator:
    def __init__(
        self,
        model,
        data_reader,
        save_data: bool = False,
        plot_figure: bool = True,
    ):
        self.model = model
        self.model.eval()
        if isinstance(data_reader, partial):
            from curator.layer.utils import find_layer_by_name_recursive
            self.data_reader = data_reader(find_layer_by_name_recursive(self.model, "cutoff"))
        else:
            self.data_reader = data_reader
        self.save_data = save_data
        self.plot_figure = plot_figure
        self.device = next(model.parameters()).device
    
    def evaluate(self, datapath):
        traj = read_trajectory(datapath)
        
        labels = defaultdict(list)
        predicts = defaultdict(list)

        head = ""
        for k in self.model.model_outputs:
            head += f'{k+'_ae':>12s}'
            head += f'{k+'_se':>12s}'

        logger.debug("# Configuration   num_atoms" + head)

        for i, atoms in enumerate(traj):
            sample = self.data_reader(atoms)
            sample = {k: v.to("cuda") for k, v in sample.items()}
            out = self.model(sample)

            # collect results
            errors = {}
            for k in self.model.model_outputs:
                label, pred = sample[k].cpu().numpy(), out[k].detach().cpu().numpy()
                labels[k].append(label)
                predicts[k].append(pred)
                errors[k + '_ae'] = np.sum(np.abs(label - pred)) / label.shape[0]
                errors[k + '_se'] = np.sum(np.square(label - pred)) / label.shape[0]
            
            error_str = ""
            for v in errors.values():
                error_str += f"{v:>12.3g}"
            logger.debug(f"{i:>15d}{len(atoms):>12d}" + error_str)

        self.labels = labels
        self.predicts = predicts
        if self.save_data:
            pickle.dump(labels, file=open('labels.pkl', 'wb'))
            pickle.dump(predicts, file=open('predicts.pkl', 'wb'))
        
        if self.plot_figure:
            self.plot()

    def plot(self):
        for k in self.labels:
            labels, predicts = np.concatenate(labels[k]).flatten(), np.concatenate(predicts[k]).flatten()
            plt.plot(self.labels[k].flatten(), self.predicts[k].flatten())
            ax = plt.gca()
            ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle='--', color='k')
            plt.xlabel('DFT')
            plt.ylabel('Prediction')
            plt.title(f"{k} prediction")
            plt.savefig(f'{k}.jpg', dpi=300)
            plt.clf()