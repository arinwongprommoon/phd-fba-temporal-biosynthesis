#!/usr/bin/env python3

import cobra
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from collections import namedtuple
from wrapt_timeout_decorator import *

# Constants needed for ablation calculations
# From Bionumbers
CELL_DRY_MASS = 15e-12  # g
# Computed from isa reactions in Yeast8 model,
# see molecular weights notebook
MW_CARB = 350.37049806068  # g/mol
MW_DNA = 3.898762476
MW_RNA = 64.04235752722991
MW_PROTEIN = 504.37442340123584
MW_COFACTOR = 4.8326512432304005
MW_ION = 2.4815607543700002
MW_LIPID = 31.5659867112958
MW_BIOMASS = 961.5662401740419

# Defaut IDs for growth and biomass reactions for batch Yeast8 model
GROWTH_ID = "r_2111"
BIOMASS_ID = "r_4041"
# IDs of pseudoreactions connected to biomass.
# Doesn't contain biomass itself (though it is in the 'Growth' subsystem)
# to make coding easier.
GROWTH_SUBSYSTEM_IDS = [
    "r_2108",
    "r_4047",
    "r_4048",
    "r_4050",
    "r_4049",
    "r_4598",
    "r_4599",
]

# List genes to delete and exchange reactions to add for each auxotroph
AuxotrophProperties = namedtuple(
    "AuxotrophProperties", ["genes_to_delete", "exch_to_add"]
)

AUXOTROPH_DICT = {
    "BY4741": AuxotrophProperties(
        ["YOR202W", "YCL018W", "YLR303W", "YEL021W"],
        [
            "r_1893",
            "r_1899",
            "r_1902",
            "r_2090",
            "r_1893_REV",
            "r_1899_REV",
            "r_1902_REV",
            "r_2090_REV",
        ],
    ),
    "BY4742": AuxotrophProperties(
        ["YOR202W", "YCL018W", "YBR115C", "YEL021W"],
        [
            "r_1893",
            "r_1899",
            "r_1900",
            "r_2090",
            "r_1893_REV",
            "r_1899_REV",
            "r_1900_REV",
            "r_2090_REV",
        ],
    ),
    "BY4741_orig": AuxotrophProperties(
        ["YOR202W", "YCL018W", "YLR303W", "YEL021W"],
        [
            "EX_his__L_e",
            "EX_met__L_e",
            "EX_leu__L_e",
            "EX_ura_e",
        ],
    ),
    "BY4742_orig": AuxotrophProperties(
        ["YOR202W", "YCL018W", "YBR115C", "YEL021W"],
        [
            "EX_his__L_e",
            "EX_met__L_e",
            "EX_lys__L_e",
            "EX_ura_e",
        ],
    ),
}

# TODO: Add lookup tables:
# - Carbon sources/other nutrients
# - Genes (systematic vs common)


class BiomassComponent:
    """
    Convenience class to store properties associated with a biomass component,
    e.g. macromolecule pseudo-metabolite

    Attributes
    ----------
    metabolite_label : string
        Name of metabolite.
    metabolite_id : string
        Metabolite ID of metabolite in cobrapy model.
    pseudoreaction : string
        Reaction ID of producing pseudoreaction in cobrapy model.
    molecular_mass : float
        Molecular mass of metabolite, in g/mol.
    ablated_flux : float
        Flux of ablated biomass reaction (to prioritise the metabolite of
        interest), to be computed, in h-1.
    est_time : float
        Doubling time estimated from flux of ablated biomass reaction, to be
        computed, in hours.
    """

    def __init__(self, metabolite_label, metabolite_id, pseudoreaction, molecular_mass):
        self.metabolite_label = metabolite_label
        self.metabolite_id = metabolite_id
        self.pseudoreaction = pseudoreaction
        self.molecular_mass = molecular_mass

        self.ablated_flux = None  # h-1
        self.est_time = None  # h

    def get_est_time(self):
        """Estimate doubling time based on growth rate"""
        self.est_time = (self.molecular_mass / MW_BIOMASS) * (
            np.log(2) / self.ablated_flux
        )


Lipids = BiomassComponent(
    metabolite_label="lipid",
    metabolite_id="s_1096[c]",
    pseudoreaction="r_2108",
    molecular_mass=MW_LIPID,
)

Proteins = BiomassComponent(
    metabolite_label="protein",
    metabolite_id="s_3717[c]",
    pseudoreaction="r_4047",
    molecular_mass=MW_PROTEIN,
)

Carbohydrates = BiomassComponent(
    metabolite_label="carbohydrate",
    metabolite_id="s_3718[c]",
    pseudoreaction="r_4048",
    molecular_mass=MW_CARB,
)

DNA = BiomassComponent(
    metabolite_label="DNA",
    metabolite_id="s_3720[c]",
    pseudoreaction="r_4050",
    molecular_mass=MW_DNA,
)

RNA = BiomassComponent(
    metabolite_label="RNA",
    metabolite_id="s_3719[c]",
    pseudoreaction="r_4049",
    molecular_mass=MW_RNA,
)

Cofactors = BiomassComponent(
    metabolite_label="cofactor",
    metabolite_id="s_4205[c]",
    pseudoreaction="r_4598",
    molecular_mass=MW_COFACTOR,
)

Ions = BiomassComponent(
    metabolite_label="ion",
    metabolite_id="s_4206[c]",
    pseudoreaction="r_4599",
    molecular_mass=MW_ION,
)

# original, non-enzyme constrained model
# defined because it annoyingly uses different naming conventions

Lipids_orig = BiomassComponent(
    metabolite_label="lipid",
    metabolite_id="s_1096",
    pseudoreaction="r_2108",
    molecular_mass=MW_LIPID,
)

Proteins_orig = BiomassComponent(
    metabolite_label="protein",
    metabolite_id="s_3717",
    pseudoreaction="r_4047",
    molecular_mass=MW_PROTEIN,
)

Carbohydrates_orig = BiomassComponent(
    metabolite_label="carbohydrate",
    metabolite_id="s_3718",
    pseudoreaction="r_4048",
    molecular_mass=MW_CARB,
)

DNA_orig = BiomassComponent(
    metabolite_label="DNA",
    metabolite_id="s_3720",
    pseudoreaction="r_4050",
    molecular_mass=MW_DNA,
)

RNA_orig = BiomassComponent(
    metabolite_label="RNA",
    metabolite_id="s_3719",
    pseudoreaction="r_4049",
    molecular_mass=MW_RNA,
)

Cofactors_orig = BiomassComponent(
    metabolite_label="cofactor",
    metabolite_id="s_4205",
    pseudoreaction="r_4598",
    molecular_mass=MW_COFACTOR,
)

Ions_orig = BiomassComponent(
    metabolite_label="ion",
    metabolite_id="s_4206",
    pseudoreaction="r_4599",
    molecular_mass=MW_ION,
)

biomass_component_list_orig = [
    Lipids_orig,
    Proteins_orig,
    Carbohydrates_orig,
    DNA_orig,
    RNA_orig,
    Cofactors_orig,
    Ions_orig,
]


class Yeast8Model:
    """
    Yeast8-derived model (cobrapy) with functions for modification.

    Attributes
    ----------
    model_filepath : string
        Filepath of model.  Should refer to a SBML-formatted .xml file or a
        YAML (.yml) file.
    model : cobra.Model object
        Model.
    growth_id : string
        Reaction ID of growth reaction.
    biomass_id : string
        Reaction ID of biomass reaction.
    biomass_component_list : list of BiomassComponent objects
        List of biomass components in the growth subsystem of the model.
    solution : cobra.Solution object
        Optimisation (FBA) solution of model.
    auxotrophy : string
        Name of auxotrophic background strain, if applicable.
    deleted_genes : list of string
        List of genes deleted in model, beyond auxotrophy, if applicable.
    ablation_result : pandas.DataFrame object
        Results of ablation study.  Columns: 'priority component' (biomass
        component being prioritised), 'flux' (flux of ablated biomass reaction),
        'est_time' (estimated doubling time based on flux).  Rows: 'original'
        (un-ablated biomass), other rows indicate biomass component.
    """

    def __init__(self, model_filepath, growth_id=GROWTH_ID, biomass_id=BIOMASS_ID):
        self.model_filepath = model_filepath
        # Load wild-type model
        if model_filepath.endswith(".xml"):
            self.model = cobra.io.read_sbml_model(model_filepath)
        elif model_filepath.endswith(".yml"):
            self.model = cobra.io.load_yaml_model(model_filepath)
        else:
            raise Exception(
                "Invaild file format for model. Valid formats include: *.xml (SBML), *.yml (YAML)."
            )
        # Copy, store as model from input file, to be used by reset method
        self.model_fromfile = self.model.copy()
        # Space to store model in intermediate steps, to be used by checkpoint
        # and reset_to_checkpoint methods
        self.model_saved = None

        self.growth_id = growth_id
        self.biomass_id = biomass_id
        # Unrestrict growth
        self.unrestrict_growth()
        print(f"Growth ({growth_id}) unrestricted.")

        # Biomass components
        self.biomass_component_list = [
            Lipids,
            Proteins,
            Carbohydrates,
            DNA,
            RNA,
            Cofactors,
            Ions,
        ]

        # TODO: getters/setters?
        self.solution = None
        self.auxotrophy = None
        # TODO: Validation of deleted genes?  i.e. check that bounds are truly
        # zero before defining
        self.deleted_genes = []
        self.ablation_result = None

        # For set_flux_penalty(); store data to save time.
        self._flux_penalty_sum = None
        self._penalty_coefficient = None

    def reset_to_file(self, hard=False):
        """Reset model to filepath

        Parameters
        ----------
        hard : bool
            Whether to hard-reset, i.e. re-load from file.  Useful for debugging.
            Otherwise, it resets to a saved copy generated when the object is
            instantiated.
        """
        if hard:
            print(f"Hard resetting model to file...")
            self.model = cobra.io.read_sbml_model(self.model_filepath)
            print(f"Done re-loading from file.")
        else:
            print(f"Resetting model to saved copy from file...")
            self.model = self.model_fromfile.copy()
            print(f"Done resetting.")
        print(
            f"Warning-- No guarantee that growth bounds are unrestricted.",
            f"If it is important that growth bounds are unrestricted, run:",
            f"yeast8model.unrestrict_growth()",
            sep=os.linesep,
        )

    def reset(self):
        """(Deprecated) Reset model to filepath"""
        print("Warning-- reset() method deprecated.  Use reset_to_file() instead.")
        self.reset_to_file(hard=True)

    def checkpoint_model(self):
        """Save a copy of the current model."""
        self.model_saved = self.model.copy()

    def reset_to_checkpoint(self):
        """Reset model to saved copy.  If no saved copy, resets to filepath."""
        if self.model_saved is not None:
            print(f"Resetting model to saved/checkpointed model...")
            self.model = self.model_saved.copy()
            print(f"Done resetting.")
        else:
            print(f"Warning-- No model currently saved.")
            self.reset_to_file()

    def unrestrict_growth(self):
        """Unrestrict growth reaction"""
        self.model.reactions.get_by_id(self.growth_id).bounds = (0, 1000)

    def knock_out_list(self, genes_to_delete):
        """Knock out list of genes

        Parameters
        ----------
        genes_to_delete : list of string
            List of genes to delete, as systematic gene names.
        """
        for gene_id in genes_to_delete:
            try:
                self.model.genes.get_by_id(gene_id).knock_out()
            except KeyError as e:
                print(f"Error-- Cannot knock out. Gene not found: {gene_id}")
        self.deleted_genes.append(genes_to_delete)

    def add_media_components(self, exch_to_unbound):
        """Add media components

        Parameters
        ----------
        exch_to_unbound : list of string
            List of exchange reactions (associated with nutrient components) to
            remove bounds for.
        """
        for exch_id in exch_to_unbound:
            try:
                self.model.reactions.get_by_id(exch_id).bounds = (-1000, 0)
            except KeyError as e:
                print(
                    f"Error-- Cannot add media component. Exchange reaction not found: {exch_id}"
                )

    def remove_media_components(self, exch_to_unbound):
        """Remove media components

        Parameters
        ----------
        exch_to_unbound : list of string
            List of exchange reactions (associated with nutrient components) to
            fix bounds for.  Bounds fixed to (0, 0) to remove influence of
            exchange reaction.
        """
        for exch_id in exch_to_unbound:
            try:
                self.model.reactions.get_by_id(exch_id).bounds = (0, 0)
            except KeyError as e:
                print(
                    f"Error-- Cannot remove media component. Exchange reaction not found: {exch_id}"
                )

    def make_auxotroph(self, auxo_strain, supplement_media=True):
        """Make the model an auxotrophic strain

        Make the model an auxotrophic strain, optionally supplement media
        composition to allow for strain to grow. Available strains include:
        BY4741, BY4742.

        Parameters
        ----------
        auxo_strain : string
            Name of auxotrophic strain.
        supplement_media : bool
            If true, add exchange reactions to simulate adding supplements
            needed for chosen auxotrophic strain to grow.

        Raises
        ------
        Exception
            Error raised if supplied auxotrophic strain is not in list of valid
            strains.
        """
        if self.auxotrophy is not None:
            # This warning is necessary because running successive rounds of
            # this method will delete the union of genes to be deleted among all
            # auxotrophies applied to model so far.
            print(
                f"Warning-- strain has existing auxotrophy: {self.auxotrophy}",
                f"Invoking make_auxotroph may cause unintended effects.",
                f"For best practise, reset the model to its source file (Yeast8Model.reset_to_file())",
                f"before proceeding.",
                sep=os.linesep,
            )
        if auxo_strain in AUXOTROPH_DICT.keys():
            # Knock out genes to create auxotroph
            self.knock_out_list(AUXOTROPH_DICT[auxo_strain].genes_to_delete)
            # By default, supplements nutrients
            if supplement_media:
                self.add_media_components(AUXOTROPH_DICT[auxo_strain].exch_to_add)
            self.auxotrophy = auxo_strain
        else:
            raise Exception(
                f"Invalid string for auxotroph strain background: {auxo_strain}"
            )

    def set_flux_penalty(self, penalty_coefficient=0.0):
        """Add a penalty to the objective function proportional to the sum of squares of fluxes

        Add a penalty to the objective function, proportional to the sum of
        squares of fluxes. The penalty coefficient is supplied by the user.
        This method relies on the proprietary ($$$) Gurobi solver, and usually
        takes a couple minutes to run.

        Parameters
        ----------
        penalty_coefficient : float
            Penalty coefficient, default 0 (i.e. no penalty applied).

        Examples
        --------
        # Instantiate model object
        y = Yeast8Model("./models/yeast-GEM_8-6-0.xml")

        # Set flux penalty
        y.set_flux_penalty(penalty_coefficient=0.1)

        # Optimize and store solution
        sol_pen = y.optimize()
        """
        self.model.solver = "gurobi"

        # Reactions to exclude needs to be hard-coded in GROWTH_SUBSYSTEM_IDS
        # because they aren't conveniently labelled
        # as part of the 'Growth' subsystem in the non-ec model.
        reactions_to_exclude = GROWTH_SUBSYSTEM_IDS + [self.growth_id, self.biomass_id]
        non_biomass_reactions = self.model.reactions.query(
            lambda x: x.id not in reactions_to_exclude
        )
        # Define expression for objective function.
        # This value is ADDED to the existing objective that has growth already
        # defined.

        # TODO: Speed this up even more
        # Re-using possible because I don't expect the flux expression to change
        # as there are no methods to add or erase reactions.
        if self._flux_penalty_sum is None:
            print("Defining flux penalty sum for the first time.")
            print("Allow a couple minutes...")
            reaction_flux_expressions = np.array(
                [reaction.flux_expression for reaction in non_biomass_reactions],
                dtype="object",
            )
            flux_penalty_sum = np.sum(np.square(reaction_flux_expressions))
            self._flux_penalty_sum = flux_penalty_sum
        else:
            print("Re-using flux penalty sum.")
            flux_penalty_sum = self._flux_penalty_sum
        flux_penalty_expression = penalty_coefficient * flux_penalty_sum

        # Set the objective.
        flux_penalty_objective = self.model.problem.Objective(
            flux_penalty_expression, direction="min"
        )
        self.model.objective = flux_penalty_objective
        # User then uses the optimize() method below to solve it.

        # Save penalty coefficient, useful for ablate()
        self._penalty_coefficient = penalty_coefficient

    def optimize(self, model=None, timeout_time=60):
        # Unlike previous methods, takes a model object as input because I need
        # to re-use this in ablate().
        """Simulate model with FBA

        Parameters
        ----------
        model : cobra.Model object
            Model.  If this is not specified, uses the class's model attribute.
        timeout_time : int
            Time till timeout in seconds. Default 60.

        Returns
        -------
        cobra.Solution object
            Solution of model simulation.

        Examples
        --------
        y = Yeast8Model('./path/to/model.xml')
        y.optimize()       # simulates y.model
        y.optimize(m)      # simulates model m, defined elsewhere
        sol = y.optimize() # saves output somewhere
        """

        @timeout(timeout_time)
        def _optimize_internal(model):
            return model.optimize()

        try:
            if model is None:
                model = self.model
            solution = _optimize_internal(model)
            return solution
        except TimeoutError as e:
            print(f"Model optimisation timeout, {timeout_time} s")

    def ablate(self, input_model=None):
        """Ablate biomass components and get growth rates & doubling times

        Ablate components in biomass reaction (i.e. macromolecules like lipids,
        proteins, carbohydrates, DNA, RNA, and also cofactors & ions) that have
        pseudoreactions associated with the 'Growth' subsystem, in turn. This
        means changing the stoichiometric matrix. In each round, the model is
        simulated and the flux is recorded. Doubling time is computed, taking
        into account the mass fraction of each biomass component -- i.e. if the
        component is a smaller fraction of the cell, it takes less time.

        Parameters
        ----------
        input.model : cobra.Model object, optional
            Input model.  If not specified, use the one associated with the
            object.

        Returns
        -------
        pandas.DataFrame object
            Results of ablation study.  Columns: 'priority component' (biomass
            component being prioritised), 'ablated_flux' (flux of ablated
            biomass reaction), 'ablated_est_time' (estimated doubling time based
            on flux), 'proportional_est_time' (estimated biomass synthesis time,
            proportional to mass fraction).  Rows: 'original' (un-ablated
            biomass), other rows indicate biomass component.
        """
        if input_model is None:
            # Copy model -- needed to restore the un-ablated model to work with
            # in successive loops
            model_working = self.model.copy()
        else:
            model_working = input_model

        # UN-ABLATED
        fba_solution = self.optimize(model_working)
        original_flux = fba_solution.fluxes[self.growth_id]
        original_est_time = np.log(2) / original_flux
        # ABLATED
        # Set up lists
        all_metabolite_ids = [
            biomass_component.metabolite_id
            for biomass_component in self.biomass_component_list
        ]
        all_pseudoreaction_ids = [
            (biomass_component.metabolite_label, biomass_component.pseudoreaction)
            for biomass_component in self.biomass_component_list
        ]
        all_pseudoreaction_ids.append(("biomass", BIOMASS_ID))
        all_pseudoreaction_ids.append(("objective", self.growth_id))
        # Loop
        for biomass_component in self.biomass_component_list:
            # boilerplate: lookup
            to_ablate = all_metabolite_ids.copy()
            to_ablate.remove(biomass_component.metabolite_id)
            to_ablate_keys = [
                model_working.metabolites.get_by_id(metabolite_id)
                for metabolite_id in to_ablate
            ]
            to_ablate_dict = dict(zip(to_ablate_keys, [-1] * len(to_ablate_keys)))

            # ablate metabolites from biomass reaction
            model_working.reactions.get_by_id(self.biomass_id).subtract_metabolites(
                to_ablate_dict
            )
            # optimise model
            fba_solution = model_working.optimize()
            # store outputs
            biomass_component.ablated_flux = fba_solution.fluxes[self.growth_id]
            biomass_component.get_est_time()

            # restore metabolites after ablation
            # Using this rather than defining a variable to restore values to
            # because keys of metabolites dict are objects with addresses.
            model_working.reactions.get_by_id(self.biomass_id).add_metabolites(
                to_ablate_dict
            )
        # construct output dataframe
        d = {
            "priority_component": ["original"]
            + [
                biomass_component.metabolite_label
                for biomass_component in self.biomass_component_list
            ],
            # Flux through ablated biomass reactions
            "ablated_flux": [original_flux]
            + [
                biomass_component.ablated_flux
                for biomass_component in self.biomass_component_list
            ],
            # Estimated doubling time, taking into account the ablated content
            # of the virtual cell's biomass
            "ablated_est_time": [original_est_time]
            + [
                biomass_component.est_time
                for biomass_component in self.biomass_component_list
            ],
            # Estimated time for each biomass component, assuming that it is
            # proportional to mass fraction
            "proportional_est_time": [original_est_time]
            + [
                (biomass_component.molecular_mass / MW_BIOMASS)
                * (np.log(2) / original_flux)
                for biomass_component in self.biomass_component_list
            ],
        }
        return pd.DataFrame(data=d)

    def ablation_barplot(self, ax, ablation_result=None):
        """Draws bar plot showing synthesis times from ablation study

        Parameters
        ----------
        ax : matplotlib.pyplot.Axes object
            Axes to draw bar plot on.
        ablation_result : pandas.DataFrame object
            Results of ablation study.  Columns: 'priority component' (biomass
            component being prioritised), 'ablated_flux' (flux of ablated
            biomass reaction), 'ablated_est_time' (estimated doubling time based
            on flux), 'proportional_est_time' (estimated biomass synthesis time,
            proportional to mass fraction).  Rows: 'original' (un-ablated
            biomass), other rows indicate biomass component.

        Examples
        --------
        # Initialise model
        y = Yeast8Model('./path/to/model.xml')

        # Ablate
        y.ablation_result = y.ablate()

        # Draw bar plot
        fig, ax = plt.subplots()
        y.ablation_barplot(ax)
        plt.show()
        """
        # Default argument
        if ablation_result is None:
            ablation_result = self.ablation_result
        # Check if ablation already done.  If not, then the value should still
        # be None despite above if statement.  If ablation already done, draw.
        if ablation_result is not None:
            # Get values for each bar plot series from ablation result DataFrame
            values_ablated, values_proportion = _bar_vals_from_ablation_df(
                ablation_result
            )

            # Draw bar plot
            # https://www.python-graph-gallery.com/8-add-confidence-interval-on-barplot
            barwidth = 0.4

            bar_labels = ablation_result.priority_component.to_list()
            bar_labels[0] = "all biomass"

            x_ablated = np.arange(len(bar_labels))
            x_proportion = [x + barwidth for x in x_ablated]

            ax.bar(
                x=x_ablated,
                height=values_ablated,
                width=barwidth,
                color="#3714b0",
                label="From ablating components\n in the biomass reaction",
            )
            ax.bar(
                x=x_proportion,
                height=values_proportion,
                width=barwidth,
                color="#cb0077",
                label="From mass fractions\n of each biomass component",
            )
            ax.set_xticks(
                ticks=[x + barwidth / 2 for x in range(len(x_ablated))],
                labels=bar_labels,
                rotation=45,
            )
            ax.set_xlabel("Biomass component")
            ax.set_ylabel("Estimated synthesis time (hours)")
            ax.legend()
        else:
            print(
                "No ablation result. Please run ablate() to generate results before plotting."
            )

    def ablation_grid(self, exch_rate_dict):
        """Array of ablation ratios from varying two exchange reactions

        Parameters
        ----------
        exch_rate_dict : dict
            dict that stores the two exchange reactions to vary and the uptake
            rate values to use.  It should be in this format:

            d = {
                'r_exch_rxn_1' : <array-like>,
                'r_exch_rxn_2' : <array-like>,
                }

        Returns
        -------
        ablation_result_array : 2-dimensional numpy.ndarray of objects
            Array of ablation result DataFrames.
            Indexing follows the exchange reaction
            arrays in the input exch_rate_dict, i.e. ratio_array[x][y]
            corresponds to exch_rate_dict['r_exch_rxn_1'][x] and
            exch_rate_dict['r_exch_rxn_2'][y].

        Examples
        --------
        # Instantiate model
        y = Yeast8Model("./models/ecYeastGEM_batch_8-6-0.xml")

        # Define exchange rate dict
        exch_rate_dict = {
            "r_1714": np.linspace(0, 18, 2),
            "r_1654": np.linspace(0, 18, 2),
        }

        # Construct array
        ara = y.ablation_grid(exch_rate_dict)
        """
        # TODO: Don't overwrite model-saved
        print(
            f"Warning: Saving current state of model to model_saved attribute.",
            f"Existing saved model will be overwritten.",
            sep=os.linesep,
        )
        self.checkpoint_model()

        # Check that the dict input argument is the right format, i.e.
        # two items.
        # string, then array-like.
        # And checks that the strings are reactions present in the model.
        # TODO: Add code to do that here

        # simplify dict syntax
        exch1_id = list(exch_rate_dict.keys())[0]
        exch2_id = list(exch_rate_dict.keys())[1]
        exch1_fluxes = list(exch_rate_dict.values())[0]
        exch2_fluxes = list(exch_rate_dict.values())[1]
        # define output array
        x_dim = len(exch1_fluxes)
        y_dim = len(exch2_fluxes)
        ablation_result_array = np.zeros(shape=(x_dim, y_dim), dtype="object")

        for x_index, exch1_flux in enumerate(exch1_fluxes):
            for y_index, exch2_flux in enumerate(exch2_fluxes):
                # model_working = self.model_saved.copy()
                model_working = self.model_saved
                # block glucose
                model_working.reactions.get_by_id("r_1714").bounds = (0, 0)
                try:
                    model_working.reactions.get_by_id("r_1714_REV").bounds = (0, 0)
                except KeyError as e:
                    print("r_1714_REV not found, ignoring in glucose-blocking step")
                # set bounds
                model_working.reactions.get_by_id(exch1_id).bounds = (-exch1_flux, 0)
                model_working.reactions.get_by_id(exch2_id).bounds = (-exch2_flux, 0)
                # deal with reversible exchange reactions, with
                # error handling in case these reactions don't exist
                try:
                    exch1_id_rev = exch1_id + "_REV"
                    model_working.reactions.get_by_id(exch1_id_rev).bounds = (
                        0,
                        exch1_flux,
                    )
                except KeyError as e:
                    print(
                        f"Error-- reversible exchange reaction {exch1_id_rev} not found. Ignoring."
                    )
                try:
                    exch2_id_rev = exch2_id + "_REV"
                    model_working.reactions.get_by_id(exch2_id_rev).bounds = (
                        0,
                        exch2_flux,
                    )
                except KeyError as e:
                    print(
                        f"Error-- reversible exchange reaction {exch2_id_rev} not found. Ignoring."
                    )

                ablation_result = self.ablate(input_model=model_working)
                ablation_result_array[x_index, y_index] = ablation_result

        return ablation_result_array


def compare_fluxes(ymodel1, ymodel2):
    """Compare fluxes between two models

    Compare fluxes between two models.  If fluxes aren't already computed and
    stored, run simulations first.

    Parameters
    ----------
    ymodel1 : Yeast8Model object
        Model 1.
    ymodel2 : Yeast8Model object
        Model 2.

    Returns
    -------
    pandas.DataFrame object
        Fluxes, sorted by magnitude, large changes on top.  Indices show
        reaction ids, values show the fluxes (with signs, positive or negative)

    Examples
    --------
    # Initialise model-handling objects
    y = Yeast8Model("./models/ecYeastGEM_batch.xml")
    z = Yeast8Model("./models/ecYeastGEM_batch.xml")

    # Make z different
    z.make_auxotroph("BY4741")

    # Compare fluxes
    dfs = compare_fluxes(y, z)
    """
    # Check if fluxes have already been computed.
    # If not, compute automatically.
    for idx, ymodel in enumerate([ymodel1, ymodel2]):
        if ymodel.solution is None:
            print(f"Model {idx+1} has no stored solution, optimising...")
            ymodel.solution = ymodel.optimize()

    diff_fluxes = ymodel2.solution.fluxes - ymodel1.solution.fluxes
    nonzero_idx = diff_fluxes.to_numpy().nonzero()[0]
    diff_fluxes_nonzero = diff_fluxes[nonzero_idx]
    # Sort by absolute flux value, large changes on top.
    diff_fluxes_sorted = diff_fluxes_nonzero[
        diff_fluxes_nonzero.abs().sort_values(ascending=False).index
    ]

    return diff_fluxes_sorted


def get_exch_saturation(ymodel, exch_id, exch_rates, remove_glucose=True):
    """Get exchange reaction saturation curve

    Get exchange reaction saturation curve. Varies exchange reaction uptake
    value and optimises for growth at each uptake value.

    Parameters
    ----------
    ymodel : Yeast8Model object
        Model.
    exch_id : string
        Reaction ID of exchange reaction.
    exch_rates : array-like
        List of uptake values to use.
    remove_glucose : bool
        Whether to remove glucose from media.  Useful if investigating carbon
        sources.

    Examples
    --------
    y = Yeast8Model("./models/ecYeastGEM_batch_8-6-0.xml")
    glc_exch_rates = np.linspace(0, 18.6, 10)
    grs = get_exch_saturation(y, "r_1714", glc_exch_rates)

    import matplotlib.pyplot as plt
    plt.plot(glc_exchrates, grs)
    """
    # Check for _REV rxn
    exch_id_rev = exch_id + "_REV"
    try:
        exch_rev = ymodel.model.reactions.get_by_id(exch_id_rev)
        print(
            f"Reversible exchange reaction {exch_id_rev} found and taken into account."
        )
        exch_rev_present = True
    except KeyError as e:
        print(
            f"Error-- reversible exchange reaction {exch_id_rev} not found. Ignoring."
        )
        exch_rev_present = False

    # Kill glucose
    if remove_glucose:
        ymodel.remove_media_components(["r_1714", "r_1714_REV"])

    growthrates = []
    for exch_rate in exch_rates:
        # negative due to FBA conventions re exchange reactions
        ymodel.model.reactions.get_by_id(exch_id).bounds = (-exch_rate, 0)
        # positive due to FBA conventions re reversible reaction
        if exch_rev_present:
            exch_rev.bounds = (0, exch_rate)
        ymodel.solution = ymodel.optimize()
        growthrates.append(ymodel.solution.fluxes[ymodel.growth_id])
    return growthrates


def compare_ablation_times(ablation_result1, ablation_result2, ax):
    """Compare two ablation study results

    Compare two ablation study results. Computes fold change of times (second
    study relative to the first), and draws on a bar plot on a log2 scale.

    Parameters
    ----------
    ablation_result1 : pandas.DataFrame object
        Results of ablation study.  Columns: 'priority component' (biomass
        component being prioritised), 'ablated_flux' (flux of ablated
        biomass reaction), 'ablated_est_time' (estimated doubling time based
        on flux), 'proportional_est_time' (estimated biomass synthesis time,
        proportional to mass fraction).  Rows: 'original' (un-ablated
        biomass), other rows indicate biomass component.
    ablation_result2 : pandas.DataFrame object
        Same as ablation_result1.  ablation_result2 is compared against
        ablation_result1.
    ax : matplotlib.pyplot.Axes object
        Axes to draw bar plot on.

    Examples
    --------
    # Initialise model-handling objects
    y = Yeast8Model("./models/ecYeastGEM_batch.xml")
    z = Yeast8Model("./models/ecYeastGEM_batch.xml")

    # Make z different
    z.make_auxotroph("BY4741")

    # Ablate
    y.ablation_result = y.ablate()
    z.ablation_result = z.ablate()

    # Compare ablation times
    fig, ax = plt.subplots()
    compare_ablation_times(z.ablation_result, y.ablation_result, ax)
    plt.show()
    """
    # Compute fold changes
    values_ablated1, values_proportion1 = _bar_vals_from_ablation_df(ablation_result1)
    values_ablated2, values_proportion2 = _bar_vals_from_ablation_df(ablation_result2)

    foldchange_ablated = np.array(values_ablated2) / np.array(values_ablated1)
    foldchange_proportion = np.array(values_proportion2) / np.array(values_proportion1)

    # Draw bar plot
    barwidth = 0.4
    bar_labels = ablation_result1.priority_component.to_list()
    bar_labels[0] = "all biomass"
    x_ablated = np.arange(len(bar_labels))
    x_proportion = [x + barwidth for x in x_ablated]
    ax.bar(
        x=x_ablated,
        height=np.log2(foldchange_ablated),
        width=barwidth,
        color="#3714b0",
        label="From ablating components\n in the biomass reaction",
    )
    ax.bar(
        x=x_proportion,
        height=np.log2(foldchange_proportion),
        width=barwidth,
        color="#cb0077",
        label="From mass fractions\n of each biomass component",
    )
    ax.set_xticks(
        ticks=[x + barwidth / 2 for x in range(len(x_ablated))],
        labels=bar_labels,
        rotation=45,
    )
    ax.set_xlabel("Biomass component")
    ax.set_ylabel("log2 fold change of estimated time")
    ax.legend()


def get_ablation_ratio(ablation_result):
    """Get ratio to represent ablation study

    Get ratio between sum of times from ablation and longest time from
    proportional estimation, as a summary of ablation study.

    Parameters
    ----------
    ablation_result : pandas.DataFrame object
        Results of ablation study.  Columns: 'priority component' (biomass
        component being prioritised), 'ablated_flux' (flux of ablated
        biomass reaction), 'ablated_est_time' (estimated doubling time based
        on flux), 'proportional_est_time' (estimated biomass synthesis time,
        proportional to mass fraction).  Rows: 'original' (un-ablated
        biomass), other rows indicate biomass component.

    Examples
    --------
    FIXME: Add docs.

    """
    ratio, _ = _get_ablation_ratio_component(ablation_result)
    return ratio


def get_ablation_largest_component(ablation_result):
    """Get largest component from proportional estimation in ablation study


    Parameters
    ----------
    ablation_result : pandas.DataFrame object
        Results of ablation study.  Columns: 'priority component' (biomass
        component being prioritised), 'ablated_flux' (flux of ablated
        biomass reaction), 'ablated_est_time' (estimated doubling time based
        on flux), 'proportional_est_time' (estimated biomass synthesis time,
        proportional to mass fraction).  Rows: 'original' (un-ablated
        biomass), other rows indicate biomass component.

    Examples
    --------
    FIXME: Add docs.

    """
    _, largest_component = _get_ablation_ratio_component(ablation_result)
    return largest_component


@np.vectorize
def vget_ablation_ratio(ablation_result_array):
    """Get ratio to represent ablation study, apply to an array

    Get ratio between sum of times from ablation and longest time from
    proportional estimation, as a summary of ablation study.

    This is a vectorised version of get_ablation_ratio(), for convenience.

    Parameters
    ----------
    ablation_result_array : 2-dimensional numpy.ndarray of objects
        Array of ablation result DataFrames.

    Examples
    --------
    FIXME: Add docs.

    """
    return get_ablation_ratio(ablation_result_array)


@np.vectorize
def vget_ablation_largest_component(ablation_result_array):
    """Get largest component from proportional estimation, apply to an array

    Get largest component from proportional estimation in ablation study

    This is a vectorised version of get_ablation_largest_component(), for
    convenience.

    Parameters
    ----------
    ablation_result_array : 2-dimensional numpy.ndarray of objects
        Array of ablation result DataFrames.

    Examples
    --------
    FIXME: Add docs.

    """
    return get_ablation_largest_component(ablation_result_array)


def _get_ablation_ratio_component(ablation_result):
    # sum of times (ablated)
    sum_of_times = ablation_result.loc[
        ablation_result.priority_component != "original",
        ablation_result.columns == "ablated_est_time",
    ].sum()
    # get element
    sum_of_times = sum_of_times[0]

    # largest proportional_est_time, apart from original.

    # Creates reduced DataFrame that shows both priority_component and
    # proportional_est_time because I want to take note which
    # priority_component is max (in case it's not always the same).
    proportional_est_time_red = ablation_result.loc[
        ablation_result.priority_component != "original",
        ["priority_component", "proportional_est_time"],
    ]
    largest_prop_df = proportional_est_time_red.loc[
        proportional_est_time_red["proportional_est_time"].idxmax()
    ]
    largest_prop_time = largest_prop_df.proportional_est_time
    largest_prop_component = largest_prop_df.priority_component

    ratio = sum_of_times / largest_prop_time
    return ratio, largest_prop_component


def heatmap_ablation_grid(
    ax,
    exch_rate_dict,
    ratio_array,
    largest_component_array=None,
    percent_saturation=False,
    vmin=0,
    vmax=2,
    cbar_label="ratio",
):
    """Draw heatmap from 2d ablation grid

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes object
        Axes to draw heatmap on.
    exch_rate_dict : dict
        dict that stores the two exchange reactions to vary and the uptake
        rate values to use.  It should be in this format:

        d = {
            'r_exch_rxn_1' : <array-like>,
            'r_exch_rxn_2' : <array-like>,
            }

    ratio_array : numpy.ndarray (2-dimensional)
        Array of ablation ratios, output from ablation_grid()
    largest_component_array : numpy.ndarray (2-dimensional), optional
        Array of largest biomass components, output from ablation_grid()
    percent_saturation : bool, optional
        Whether to scale axis labels so that the numbers displayed are percent
        of the highest value of the axis (usually saturation).  Default False.
    vmin : float, optional
        Minimum of range for colour bar.  Default 0.
    vmax : float, optional
        Maximum of range for colour bar.  Default 2.
    cbar_label : string, optional
        Label for colour bar.  Default "ratio".

    Examples
    --------
    FIXME: Add docs.

    """
    # If largest_component_array is supplied, use it as text labels on heatmap.
    # This design takes advantage of seaborn.heatmap(annot=None) being default.
    if largest_component_array is None:
        annot_input = largest_component_array
    # TODO: Improve error-handling by checking if this is a 2D numpy array
    else:
        annot_input = np.rot90(largest_component_array)

    heatmap_xticklabels = list(exch_rate_dict.values())[0]
    heatmap_yticklabels = list(exch_rate_dict.values())[1][::-1]
    if percent_saturation:
        heatmap_xticklabels /= np.max(heatmap_xticklabels)
        heatmap_xticklabels *= 100
        heatmap_yticklabels /= np.max(heatmap_yticklabels)
        heatmap_yticklabels *= 100

    # Draws heatmap.
    # Rounding directly on the x/yticklabels variables because of known
    # matplotlib-seaborn bug:
    # - https://github.com/mwaskom/seaborn/issues/1005
    # - https://stackoverflow.com/questions/63964006/round-decimal-places-seaborn-heatmap-labels
    # - https://stackoverflow.com/questions/50571592/matplotlib-formatstrformatter-returns-wrong-values
    sns.heatmap(
        data=np.rot90(ratio_array),
        annot=annot_input,
        xticklabels=np.around(heatmap_xticklabels, decimals=3),
        yticklabels=np.around(heatmap_yticklabels, decimals=3),
        vmin=vmin,
        vmax=vmax,
        cmap="RdBu_r",
        cbar_kws={"label": cbar_label},
        fmt="",
        ax=ax,
    )
    ax.set_xlabel(list(exch_rate_dict.keys())[0])
    ax.set_ylabel(list(exch_rate_dict.keys())[1])


def piechart_ablation_grid(
    exch_rate_dict,
    ablation_result_array,
    percent_saturation=False,
    xlabel=None,
    ylabel=None,
):
    """Grid of pie charts showing proportions of prioritised components in ablated-predicted time

    Draws a grid of pie charts.  Each pie chart shows the proportions of the
    times predicted for prioritising each biomass component by ablation study.
    x and y axes show the corresponding exchange reaction fluxes.

    If a pie chart at any position cannot be drawn -- e.g. when exchange rate is
    0 -- then that position shows the text 'N/A'.

    Parameters
    ----------
    exch_rate_dict : dict
        dict that stores the two exchange reactions to vary and the uptake
        rate values to use.  It should be in this format:

        d = {
            'r_exch_rxn_1' : <array-like>,
            'r_exch_rxn_2' : <array-like>,
            }

    ablation_result_array : 2-dimensional numpy.ndarray of objects
        Array of ablation result DataFrames.
        Indexing follows the exchange reaction
        arrays in the input exch_rate_dict, i.e. ratio_array[x][y]
        corresponds to exch_rate_dict['r_exch_rxn_1'][x] and
        exch_rate_dict['r_exch_rxn_2'][y].
    percent_saturation : bool, optional
        Whether to scale axis labels so that the numbers displayed are percent
        of the highest value of the axis (usually saturation).  Default False.
    xlabel : str
        x-axis label.  Defaults to name of first exchange reaction.
    ylabel : str
        y-axis label.  Defaults to name of second exchange reaction.

    Examples
    --------
    # Initialise model
    wt_ec = Yeast8Model(...)

    # Create ablation grid
    exch_rate_dict = {
        "r_1714": np.linspace(0, 2 * 8.45, 3),  # glucose
        "r_1654": np.linspace(0, 2 * 1.45, 3),  # ammonium
    }
    ablation_result_array = wt_ec.ablation_grid(exch_rate_dict)

    # Draw pie chart
    piechart_ablation_grid(exch_rate_dict, ablation_result_array)
    plt.show()
    """
    ablation_result_array = np.rot90(ablation_result_array)

    nrows, ncols = ablation_result_array.shape
    # Have exch_rate values in an extra column & row
    nrows += 1
    ncols += 1
    fig, ax = plt.subplots(nrows, ncols)

    # Define labels for legend
    # Debt: Assumes that all ablation_result DataFrames share the same format
    ablation_result_temp = ablation_result_array[0, 0]
    component_list = ablation_result_temp.priority_component.to_numpy().T
    # deletes 'original' priority component
    component_list = np.delete(component_list, 0)
    component_list = component_list.tolist()

    # Define axis labels
    global_xaxislabels = list(exch_rate_dict.values())[0]
    # Scale if specified
    global_yaxislabels = list(exch_rate_dict.values())[1][::-1]
    if percent_saturation:
        global_xaxislabels /= np.max(global_xaxislabels)
        global_xaxislabels *= 100
        global_yaxislabels /= np.max(global_yaxislabels)
        global_yaxislabels *= 100
    # Dummy value for corner position -- not used
    # (could be useful for debugging)
    global_xaxislabels = np.append([-1], global_xaxislabels)
    global_yaxislabels = np.append(global_yaxislabels, [-1])

    # Draw pie charts
    for row_idx, global_yaxislabel in enumerate(global_yaxislabels):
        # Left column reserved for exch rate 2 labels
        ax[row_idx, 0].set_axis_off()
        # Bottom left corner must be blank
        if row_idx == len(global_yaxislabels) - 1:
            pass
        else:
            # Print exch rate label
            ax[row_idx, 0].text(
                x=0.5,
                y=0.5,
                s=f"{global_yaxislabel:.3f}",
                ha="center",
                va="center",
            )
        for col_idx, global_xaxislabel in enumerate(global_xaxislabels):
            # Bottom row reserved for exch rate 1 labels
            if row_idx == len(global_yaxislabels) - 1:
                ax[row_idx, col_idx].set_axis_off()
                # Bottom left corner must be blank
                if col_idx == 0:
                    pass
                else:
                    # Print exch rate label
                    ax[row_idx, col_idx].text(
                        x=0.5,
                        y=0.5,
                        s=f"{global_xaxislabel:.3f}",
                        ha="center",
                        va="center",
                    )
            else:
                # Left column reserved for exch rate 2 labels
                if col_idx == 0:
                    pass
                else:
                    # Get times
                    ablation_result = ablation_result_array[row_idx, col_idx - 1]
                    ablation_times_df = ablation_result.loc[
                        ablation_result.priority_component != "original",
                        ablation_result.columns == "ablated_est_time",
                    ]
                    ablation_times = ablation_times_df.to_numpy().T[0]
                    # Deal with edge cases, e.g. negative values when exch rate is 0
                    try:
                        artists = ax[row_idx, col_idx].pie(ablation_times)
                    except:
                        print(f"Unable to draw pie chart at [{row_idx}, {col_idx}].")
                        ax[row_idx, col_idx].set_axis_off()
                        ax[row_idx, col_idx].text(
                            x=0.5,
                            y=0.5,
                            s="N/A",
                            ha="center",
                            va="center",
                            fontweight="bold",
                        )

    # For global axis labels: create a big subplot and hide everything except
    # for the labels
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    if xlabel is None:
        xlabel = list(exch_rate_dict.keys())[0]
    if ylabel is None:
        ylabel = list(exch_rate_dict.keys())[1]
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Legend: colour = biomass component
    fig.legend(artists[0], component_list, loc="center right")
    fig.subplots_adjust(right=0.75)


def _bar_vals_from_ablation_df(ablation_result):
    """Get values for bar plots from ablation result DataFrame

    Takes DataFrame output from Yeast8Model.ablate() and reformats the data for
    use with bar plots. Specifically, computes the sum of times predicted from
    ablating the biomass reaction.

    Parameters
    ----------
    ablation_result : pandas.DataFrame object
        Results of ablation study.  Columns: 'priority component' (biomass
        component being prioritised), 'ablated_flux' (flux of ablated
        biomass reaction), 'ablated_est_time' (estimated doubling time based
        on flux), 'proportional_est_time' (estimated biomass synthesis time,
        proportional to mass fraction).  Rows: 'original' (un-ablated
        biomass), other rows indicate biomass component.

    Returns
    -------
    values_ablated : list of float
        List of estimated times predicted from ablating biomass reaction.  First
        element is the sum of times.  Subsequent elements are estimated times
        from focusing on each biomass component in turn.
    values_proportion : list of float
        List of estimated times based on proportion of estimated doubling time
        based on mass fractions of biomass components.  First element is the
        doubling time based on growth rate (i.e. flux of un-ablated biomass
        reaction).  Subsequent elements are proportional times.
    """
    # sum of times
    sum_of_times = ablation_result.loc[
        ablation_result.priority_component != "original",
        ablation_result.columns == "ablated_est_time",
    ].sum()
    # get element
    sum_of_times = sum_of_times[0]
    # ablated
    values_ablated = ablation_result.ablated_est_time.to_list()
    values_ablated[0] = sum_of_times
    # proportion
    values_proportion = ablation_result.proportional_est_time.to_list()

    return values_ablated, values_proportion
