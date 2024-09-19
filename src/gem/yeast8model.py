#!/usr/bin/env python3

import os

import cobra
import numpy as np
import pandas as pd
from wrapt_timeout_decorator import *

from src.constants.constants import (
    AUXOTROPH_DICT,
    BIOMASS_ID,
    GROWTH_ID,
    GROWTH_SUBSYSTEM_IDS,
    MW_BIOMASS,
)
from src.data.biomasscomponent import (
    DNA,
    RNA,
    Carbohydrates,
    Cofactors,
    Ions,
    Lipids,
    Proteins,
)


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
        # Reactions to exclude needs to be hard-coded in GROWTH_SUBSYSTEM_IDS
        # because they aren't conveniently labelled
        # as part of the 'Growth' subsystem in the non-ec model.
        reactions_to_exclude = GROWTH_SUBSYSTEM_IDS + [self.growth_id, self.biomass_id]
        self._non_biomass_reactions = self.model.reactions.query(
            lambda x: x.id not in reactions_to_exclude
        )

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

        self.ablation_reaction_fluxes = dict()
        self.ablation_enzyme_fluxes = dict()

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

    def set_flux_constraint(self, upper_bound):
        """Set upper bound to sum of absolute values of fluxes

        Parameters
        ----------
        upper_bound : float
            Upper bound to sum of absolute values of fluxes

        Examples
        --------
        wt_ec = Yeast8Model("../data/gemfiles/ecYeastGEM_batch_8-6-0.xml")
        wt_ec.set_flux_constraint(upper_bound=900)
        sol = wt_ec.optimize()

        """
        # Stolen from
        # https://cobrapy.readthedocs.io/en/latest/constraints_objectives.html#Constraints
        coefficients = dict()
        for rxn in self._non_biomass_reactions:
            coefficients[rxn.forward_variable] = 1.0
            coefficients[rxn.reverse_variable] = 1.0
        constraint = self.model.problem.Constraint(0, lb=0, ub=upper_bound)
        self.model.add_cons_vars(constraint)
        self.model.solver.update()
        constraint.set_linear_coefficients(coefficients=coefficients)

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

    def ablate(self, input_model=None, enzflux_tol=1e-16):
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
        input_model : cobra.Model object, optional
            Input model.  If not specified, use the one associated with the
            object.
        enzflux_tol : float, optional
            If specified, converts enzyme usage fluxes that have magnitudes
            below this value to zero.  In theory, enzyme usage fluxes should
            never be negative; however, this sometimes occurs depending on the
            solver.  This tolerance variable corrects this.

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

        # DICT TO STORE REACTION FLUXES
        self.ablation_reaction_fluxes = dict.fromkeys(
            ["original"]
            + [
                biomass_component.metabolite_label
                for biomass_component in self.biomass_component_list
            ]
        )
        # DICT TO STORE ENZYME USAGE FLUXES
        self.ablation_enzyme_fluxes = dict.fromkeys(
            ["original"]
            + [
                biomass_component.metabolite_label
                for biomass_component in self.biomass_component_list
            ]
        )

        # UN-ABLATED
        fba_solution = self.optimize(model_working)
        original_flux = fba_solution.fluxes[self.growth_id]
        # get reaction fluxes
        self.ablation_reaction_fluxes["original"] = fba_solution.fluxes.loc[
            fba_solution.fluxes.index.str.startswith("r_")
        ]
        # get enzyme usage fluxes
        self.ablation_enzyme_fluxes["original"] = fba_solution.fluxes.loc[
            fba_solution.fluxes.index.str.startswith("draw_prot")
        ]
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
            # get reaction fluxes
            ablation_reaction_flux = fba_solution.fluxes.loc[
                fba_solution.fluxes.index.str.startswith("r_")
            ]
            self.ablation_reaction_fluxes[
                biomass_component.metabolite_label
            ] = ablation_reaction_flux
            # get enzyme usage fluxes
            ablation_enzyme_flux = fba_solution.fluxes.loc[
                fba_solution.fluxes.index.str.startswith("draw_prot")
            ]
            ablation_enzyme_flux.loc[np.abs(ablation_enzyme_flux) < enzflux_tol] = 0
            self.ablation_enzyme_fluxes[
                biomass_component.metabolite_label
            ] = ablation_enzyme_flux
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

    def ablation_grid(self, exch_rate_dict):
        """Array of ablation DataFrames from varying two exchange reactions

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
        y = Yeast8Model("../data/gemfiles/ecYeastGEM_batch_8-6-0.xml")

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

    def usgfluxes_list(self, exch_rate_points):
        """
        Example format for exch_rate_points:
        exch_rate_points = {
            "exch_ids": ["r_1714", "r_1654"],
            "exch_points": np.array([[16.89, 2.96], [1.69, 1.05]])
        }
        """
        model_working = self.model.copy()
        ablation_result_list = np.zeros(
            shape=(len(exch_rate_points["exch_points"])), dtype="object"
        )

        for point_idx, point in enumerate(exch_rate_points["exch_points"]):
            # block glucose
            model_working.reactions.get_by_id("r_1714").bounds = (0, 0)
            try:
                model_working.reactions.get_by_id("r_1714_REV").bounds = (0, 0)
            except KeyError as e:
                print("r_1714_REV not found, ignoring in glucose-blocking step")
            # set bounds
            for exch_idx, exch_id in enumerate(exch_rate_points["exch_ids"]):
                model_working.reactions.get_by_id(exch_id).bounds = (
                    -point[exch_idx],
                    0,
                )
                # deal with reversible exchange reactions
                exch_id_rev = exch_id + "_REV"
                try:
                    model_working.reactions.get_by_id(exch_id_rev).bounds = (
                        0,
                        point[exch_idx],
                    )
                except KeyError as e:
                    print(
                        f"Error-- reversible exchange reaction {exch_id_rev} not found. Ignoring."
                    )
            ablation_result = self.ablate(input_model=model_working)
            enz_use_array = np.stack(
                [df.to_numpy() for df in self.ablation_enzyme_fluxes.values()]
            )
            ablation_result_list[point_idx] = enz_use_array

        return ablation_result_list

    def usgfluxes_grid(self, exch_rate_dict):
        # TODO: Don't overwrite model-saved
        print(
            f"Warning: Saving current state of model to model_saved attribute.",
            f"Existing saved model will be overwritten.",
            sep=os.linesep,
        )
        self.checkpoint_model()

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
                enz_use_array = np.stack(
                    [df.to_numpy() for df in self.ablation_enzyme_fluxes.values()]
                )
                ablation_result_array[x_index, y_index] = enz_use_array

        return ablation_result_array
