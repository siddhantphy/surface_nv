from argparse import ArgumentParser
import netsquid as ns
import time
from netsquid.nodes.node import Node
from netsquid_nv.nv_center import NVQuantumProcessor
from netsquid_nv.magic_distributor import NVSingleClickMagicDistributor, NVDoubleClickMagicDistributor
from netsquid_nv.nv_parameter_set import compute_product_tau_decay_delta_w_from_nodephasing_number
from netsquid_physlayer.heralded_connection import MiddleHeraldedConnection, HeraldedConnection
from netsquid_physlayer.classical_connection import ClassicalConnectionWithLength
from netsquid_netconf.netconf import ComponentBuilder, netconf_generator
from netsquid.components import QuantumProgram, INSTR_MEASURE, INSTR_H


distributor_name_to_class = {"double_click_nv": NVDoubleClickMagicDistributor,
                             "single_click_nv": NVSingleClickMagicDistributor}


class NVNode(Node):
    """Node with an NV processor.
    """
    def __init__(self, name, end_node, electron_init_depolar_prob, electron_single_qubit_depolar_prob,
                 prob_error_0, prob_error_1, carbon_init_depolar_prob, carbon_z_rot_depolar_prob,
                 ec_gate_depolar_prob, electron_T1, electron_T2, carbon_T1, carbon_T2, coherent_phase,
                 initial_nuclear_phase, p_double_exc, p_fail_class_corr, photon_emission_delay, n1e,
                 std_electron_electron_phase_drift, visibility, carbon_init_duration,
                 carbon_z_rot_duration, electron_init_duration, electron_single_qubit_duration,
                 ec_two_qubit_gate_duration, measure_duration, magical_swap_gate_duration, num_positions, cutoff_time,
                 use_magical_swap=False, port_names=None, bright_state_param=None, num_communication_qubits=1,
                 sequential_one_repeater=False, **kwargs):

        product_tau_decay_delta_w = compute_product_tau_decay_delta_w_from_nodephasing_number(n1e, alpha=0.5)
        nv_processor = NVQuantumProcessor(
            num_positions=num_positions,
            electron_init_depolar_prob=electron_init_depolar_prob,
            electron_single_qubit_depolar_prob=electron_single_qubit_depolar_prob,
            prob_error_0=prob_error_0,
            prob_error_1=prob_error_1,
            carbon_init_depolar_prob=carbon_init_depolar_prob,
            carbon_z_rot_depolar_prob=carbon_z_rot_depolar_prob,
            ec_gate_depolar_prob=ec_gate_depolar_prob,
            electron_T1=electron_T1,
            electron_T2=electron_T2,
            carbon_T1=carbon_T1,
            carbon_T2=carbon_T2,
            coherent_phase=coherent_phase,
            initial_nuclear_phase=initial_nuclear_phase,
            p_double_exc=p_double_exc,
            p_fail_class_corr=p_fail_class_corr,
            photon_emission_delay=photon_emission_delay,
            tau_decay=product_tau_decay_delta_w,
            delta_w=1.,
            std_electron_electron_phase_drift=std_electron_electron_phase_drift,
            visibility=visibility,
            carbon_init_duration=carbon_init_duration,
            carbon_z_rot_duration=carbon_z_rot_duration,
            electron_init_duration=electron_init_duration,
            electron_single_qubit_duration=electron_single_qubit_duration,
            ec_two_qubit_gate_duration=ec_two_qubit_gate_duration,
            measure_duration=measure_duration,
            magical_swap_gate_duration=magical_swap_gate_duration,
            use_magical_swap=use_magical_swap)
        super().__init__(name=name, qmemory=nv_processor, port_names=port_names)

        if bright_state_param is not None:
            self.add_property("bright_state_param", bright_state_param)


def _get_magic_distributor(network):
    """Given a network object, returns the magic distributor. It is assumed there is only one (i.e. two nodes).
    """
    for connection in network.connections.values():
        if isinstance(connection, HeraldedConnection):
            return connection.magic_distributor


def implement_magic(network, config):
    """Add magic distributors to a physical network.

    To every connection in the network a magic distributor is added as attribute.
    The distributor type must be defined in the configuration file for each heralded connection, as a property of name
    "distributor". The supported distributor types can be found in the `distributor_name_to_class` dictionary, where the
    keys are the arguments that should be defined in the configuration file, and the values are the actual distributors.
    The magic distributor is given the connection as argument upon construction (so that parameters to be used in magic
    can be read from the connection).

    Parameters
    ----------
    network : :class:`netsquid.nodes.network.Network`
        Network that the network distributors should be added to.
    config : dict
        Dictionary holding component names and their properties.

    """
    distributors = {}
    for name, connection in config["components"].items():
        if "heralded_connection" in connection["type"]:
            try:
                distributors[name] = connection["properties"]["distributor"]
            except KeyError:
                raise KeyError("Magic distributor not defined for {}".format(name))

    for connection in network.connections.values():
        if isinstance(connection, HeraldedConnection):
            nodes = [port.connected_port.component for port in connection.ports.values()]
            try:
                magic_distributor = distributor_name_to_class[distributors[connection.name]](
                    nodes=nodes, heralded_connection=connection)
            except KeyError:
                raise KeyError("{} is not a supported distributor type.".format(distributors[connection.name]))
            connection.magic_distributor = magic_distributor


def setup_networks(config_file_name):
    """
    Set up network(s) according to configuration file

    Parameters
    ----------
    config_file_name : str
        Name of configuration file.

    Returns
    -------
    generator : generator
        Generator yielding network configurations
    """

    # add required components to ComponentBuilder
    ComponentBuilder.add_type(name="nv_node", new_type=NVNode)
    ComponentBuilder.add_type(name="mid_heralded_connection", new_type=MiddleHeraldedConnection)
    ComponentBuilder.add_type(name="heralded_connection", new_type=HeraldedConnection)
    ComponentBuilder.add_type(name="classical_connection", new_type=ClassicalConnectionWithLength)

    generator = netconf_generator(config_file_name)

    return generator


def run_simulation(generator, n_runs, suppress_output=False):
    """
    Runs simulation and collects data for each network configuration. Stores data in RepchainDataFrameHolder in format
    that allows for plotting at later point.

    Parameters
    ----------
    generator : generator
        Generator of network configurations
    n_runs : int
        Number of runs per data point
    suppress_output : bool
        If true, status print statements are suppressed.

    Returns
    -------
    meas_holder : :class:`netsquid_simulationtools.repchain_dataframe_holder.RepchainDataFrameHolder`
        RepchainDataFrameHolder with collected simulation data.

    """
    ns.set_qstate_formalism(ns.qubits.qformalism.QFormalism.DM)

    for objects, config in generator:
        network = objects["network"]
        implement_magic(network, config)
        magic_distributor = _get_magic_distributor(network)
        start_time_simulation = time.time()

        print(objects)
        for run in range(n_runs):
            node_ids = [node.ID for node in network.nodes.values()]

            # currently this just places an entangled state on the communication qubit of the two nodes.
            # You can add whatever you want to do here, and can also make it more complex (with logic, etc.)
            # using protocols (see NetSquid tutorial and blueprint repository for examples)
            # The simulation data is also at the time not being collected. I imagine you'll want to do that,
            # but I wasn't sure exactly what kind of thing you'd want, so I didn't add it. There's also some
            # examples I can show you.

            magic_distributor.add_delivery(memory_positions={node_ids[0]: 0, node_ids[1]: 0})


            ns.sim_run()
        simulation_time = time.time() - start_time_simulation
        if not suppress_output:
            print(f"Performed {n_runs} runs in {simulation_time:.2e} s")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('configfile', type=str, help="Configuration file name.")
    parser.add_argument('-n', '--n_runs', type=int, required=False, default=10,
                        help="Number of simulation runs per configuration. Defaults to 10.")

    args, unknown = parser.parse_known_args()
    generator = setup_networks(args.configfile)
    run_simulation(generator, args.n_runs)