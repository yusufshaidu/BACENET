class Calculator(BaseCalculator):
    """Base-class for all ASE calculators.

    A calculator must raise PropertyNotImplementedError if asked for a
    property that it can't calculate.  So, if calculation of the
    stress tensor has not been implemented, get_stress(atoms) should
    raise PropertyNotImplementedError.  This can be achieved simply by not
    including the string 'stress' in the list implemented_properties
    which is a class member.  These are the names of the standard
    properties: 'energy', 'forces', 'stress', 'dipole', 'charges',
    'magmom' and 'magmoms'.
    """

    default_parameters: Dict[str, Any] = {}
    'Default parameters'

    ignored_changes: Set[str] = set()
    'Properties of Atoms which we ignore for the purposes of cache '
    'invalidation with check_state().'

    discard_results_on_any_change = False
    'Whether we purge the results following any change in the set() method.  '
    'Most (file I/O) calculators will probably want this.'

    def __init__(
        self,
        restart=None,
        ignore_bad_restart_file=BaseCalculator._deprecated,
        label=None,
        atoms=None,
        directory='.',
        **kwargs,
    ):
        """Basic calculator implementation.

        restart: str
            Prefix for restart file.  May contain a directory. Default
            is None: don't restart.
        ignore_bad_restart_file: bool
            Deprecated, please do not use.
            Passing more than one positional argument to Calculator()
            is deprecated and will stop working in the future.
            Ignore broken or missing restart file.  By default, it is an
            error if the restart file is missing or broken.
        directory: str or PurePath
            Working directory in which to read and write files and
            perform calculations.
        label: str
            Name used for all files.  Not supported by all calculators.
            May contain a directory, but please use the directory parameter
            for that instead.
        atoms: Atoms object
            Optional Atoms object to which the calculator will be
            attached.  When restarting, atoms will get its positions and
            unit-cell updated from file.
        """
        self.atoms = None  # copy of atoms object from last calculation
        self.results = {}  # calculated properties (energy, forces, ...)
        self.parameters = None  # calculational parameters
        self._directory = None  # Initialize

        if ignore_bad_restart_file is self._deprecated:
            ignore_bad_restart_file = False
        else:
            warnings.warn(
                FutureWarning(
                    'The keyword "ignore_bad_restart_file" is deprecated and '
                    'will be removed in a future version of ASE.  Passing more '
                    'than one positional argument to Calculator is also '
                    'deprecated and will stop functioning in the future.  '
                    'Please pass arguments by keyword (key=value) except '
                    'optionally the "restart" keyword.'
                )
            )

        if restart is not None:
            try:
                self.read(restart)  # read parameters, atoms and results
            except ReadError:
                if ignore_bad_restart_file:
                    self.reset()
                else:
                    raise

        self.directory = directory
        self.prefix = None
        if label is not None:
            if self.directory == '.' and '/' in label:
                # We specified directory in label, and nothing in the diretory
                # key
                self.label = label
            elif '/' not in label:
                # We specified our directory in the directory keyword
                # or not at all
                self.label = '/'.join((self.directory, label))
            else:
                raise ValueError(
                    'Directory redundantly specified though '
                    'directory="{}" and label="{}".  '
                    'Please omit "/" in label.'.format(self.directory, label)
                )

        if self.parameters is None:
            # Use default parameters if they were not read from file:
            self.parameters = self.get_default_parameters()

        if atoms is not None:
            atoms.calc = self
            if self.atoms is not None:
                # Atoms were read from file.  Update atoms:
                if not (
                    equal(atoms.numbers, self.atoms.numbers)
                    and (atoms.pbc == self.atoms.pbc).all()
                ):
                    raise CalculatorError('Atoms not compatible with file')
                atoms.positions = self.atoms.positions
                atoms.cell = self.atoms.cell

        self.set(**kwargs)

        if not hasattr(self, 'get_spin_polarized'):
            self.get_spin_polarized = self._deprecated_get_spin_polarized
        # XXX We are very naughty and do not call super constructor!

        # For historical reasons we have a particular caching protocol.
        # We disable the superclass' optional cache.
        self.use_cache = False

    @property
    def directory(self) -> str:
        return self._directory

    @directory.setter
    def directory(self, directory: Union[str, os.PathLike]):
        self._directory = str(Path(directory))  # Normalize path.

    @property
    def label(self):
        if self.directory == '.':
            return self.prefix

        # Generally, label ~ directory/prefix
        #
        # We use '/' rather than os.pathsep because
        #   1) directory/prefix does not represent any actual path
        #   2) We want the same string to work the same on all platforms
        if self.prefix is None:
            return self.directory + '/'

        return f'{self.directory}/{self.prefix}'

    @label.setter
    def label(self, label):
        if label is None:
            self.directory = '.'
            self.prefix = None
            return

        tokens = label.rsplit('/', 1)
        if len(tokens) == 2:
            directory, prefix = tokens
        else:
            assert len(tokens) == 1
            directory = '.'
            prefix = tokens[0]
        if prefix == '':
            prefix = None
        self.directory = directory
        self.prefix = prefix

[docs]
    def set_label(self, label):
        """Set label and convert label to directory and prefix.

        Examples:

        * label='abc': (directory='.', prefix='abc')
        * label='dir1/abc': (directory='dir1', prefix='abc')
        * label=None: (directory='.', prefix=None)
        """
        self.label = label


    def get_default_parameters(self):
        return Parameters(copy.deepcopy(self.default_parameters))

    def todict(self, skip_default=True):
        defaults = self.get_default_parameters()
        dct = {}
        for key, value in self.parameters.items():
            if hasattr(value, 'todict'):
                value = value.todict()
            if skip_default:
                default = defaults.get(key, '_no_default_')
                if default != '_no_default_' and equal(value, default):
                    continue
            dct[key] = value
        return dct

[docs]
    def reset(self):
        """Clear all information from old calculation."""

        self.atoms = None
        self.results = {}


[docs]
    def read(self, label):
        """Read atoms, parameters and calculated properties from output file.

        Read result from self.label file.  Raise ReadError if the file
        is not there.  If the file is corrupted or contains an error
        message from the calculation, a ReadError should also be
        raised.  In case of succes, these attributes must set:

        atoms: Atoms object
            The state of the atoms from last calculation.
        parameters: Parameters object
            The parameter dictionary.
        results: dict
            Calculated properties like energy and forces.

        The FileIOCalculator.read() method will typically read atoms
        and parameters and get the results dict by calling the
        read_results() method."""

        self.set_label(label)


    def get_atoms(self):
        if self.atoms is None:
            raise ValueError('Calculator has no atoms')
        atoms = self.atoms.copy()
        atoms.calc = self
        return atoms

    @classmethod
    def read_atoms(cls, restart, **kwargs):
        return cls(restart=restart, label=restart, **kwargs).get_atoms()

[docs]
    def set(self, **kwargs):
        """Set parameters like set(key1=value1, key2=value2, ...).

        A dictionary containing the parameters that have been changed
        is returned.

        Subclasses must implement a set() method that will look at the
        chaneged parameters and decide if a call to reset() is needed.
        If the changed parameters are harmless, like a change in
        verbosity, then there is no need to call reset().

        The special keyword 'parameters' can be used to read
        parameters from a file."""

        if 'parameters' in kwargs:
            filename = kwargs.pop('parameters')
            parameters = Parameters.read(filename)
            parameters.update(kwargs)
            kwargs = parameters

        changed_parameters = {}

        for key, value in kwargs.items():
            oldvalue = self.parameters.get(key)
            if key not in self.parameters or not equal(value, oldvalue):
                changed_parameters[key] = value
                self.parameters[key] = value

        if self.discard_results_on_any_change and changed_parameters:
            self.reset()
        return changed_parameters


[docs]
    def check_state(self, atoms, tol=1e-15):
        """Check for any system changes since last calculation."""
        return compare_atoms(
            self.atoms,
            atoms,
            tol=tol,
            excluded_properties=set(self.ignored_changes),
        )


[docs]
    def calculate(
        self, atoms=None, properties=['energy'], system_changes=all_changes
    ):
        """Do the calculation.

        properties: list of str
            List of what needs to be calculated.  Can be any combination
            of 'energy', 'forces', 'stress', 'dipole', 'charges', 'magmom'
            and 'magmoms'.
        system_changes: list of str
            List of what has changed since last calculation.  Can be
            any combination of these six: 'positions', 'numbers', 'cell',
            'pbc', 'initial_charges' and 'initial_magmoms'.

        Subclasses need to implement this, but can ignore properties
        and system_changes if they want.  Calculated properties should
        be inserted into results dictionary like shown in this dummy
        example::

            self.results = {'energy': 0.0,
                            'forces': np.zeros((len(atoms), 3)),
                            'stress': np.zeros(6),
                            'dipole': np.zeros(3),
                            'charges': np.zeros(len(atoms)),
                            'magmom': 0.0,
                            'magmoms': np.zeros(len(atoms))}

        The subclass implementation should first call this
        implementation to set the atoms attribute and create any missing
        directories.
        """
        if atoms is not None:
            self.atoms = atoms.copy()
        if not os.path.isdir(self._directory):
            try:
                os.makedirs(self._directory)
            except FileExistsError as e:
                # We can only end up here in case of a race condition if
                # multiple Calculators are running concurrently *and* use the
                # same _directory, which cannot be expected to work anyway.
                msg = (
                    'Concurrent use of directory '
                    + self._directory
                    + 'by multiple Calculator instances detected. Please '
                    'use one directory per instance.'
                )
                raise RuntimeError(msg) from e


[docs]
    def calculate_numerical_forces(self, atoms, d=0.001):
        """Calculate numerical forces using finite difference.

        All atoms will be displaced by +d and -d in all directions."""
        from ase.calculators.test import numeric_forces

        return numeric_forces(atoms, d=d)


[docs]
    def calculate_numerical_stress(self, atoms, d=1e-6, voigt=True):
        """Calculate numerical stress using finite difference."""
        from ase.calculators.test import numeric_stress

        return numeric_stress(atoms, d=d, voigt=voigt)


    def _deprecated_get_spin_polarized(self):
        msg = (
            'This calculator does not implement get_spin_polarized().  '
            'In the future, calc.get_spin_polarized() will work only on '
            'calculator classes that explicitly implement this method or '
            'inherit the method via specialized subclasses.'
        )
        warnings.warn(msg, FutureWarning)
        return False

[docs]
    def band_structure(self):
        """Create band-structure object for plotting."""
        from ase.spectrum.band_structure import get_band_structure

        # XXX This calculator is supposed to just have done a band structure
        # calculation, but the calculator may not have the correct Fermi level
        # if it updated the Fermi level after changing k-points.
        # This will be a problem with some calculators (currently GPAW), and
        # the user would have to override this by providing the Fermi level
        # from the selfconsistent calculation.
        return get_band_structure(calc=self)



class OldShellProfile:
    def __init__(self, command):
        self.command = command
        self.configvars = {}

    def execute(self, calc):
        if self.command is None:
            raise EnvironmentError(
                'Please set ${} environment variable '.format(
                    'ASE_' + self.calc.upper() + '_COMMAND'
                )
                + 'or supply the command keyword'
            )
        command = self.command
        if 'PREFIX' in command:
            command = command.replace('PREFIX', calc.prefix)

        try:
            proc = subprocess.Popen(command, shell=True, cwd=calc.directory)
        except OSError as err:
            # Actually this may never happen with shell=True, since
            # probably the shell launches successfully.  But we soon want
            # to allow calling the subprocess directly, and then this
            # distinction (failed to launch vs failed to run) is useful.
            msg = f'Failed to execute "{command}"'
            raise EnvironmentError(msg) from err

        errorcode = proc.wait()

        if errorcode:
            path = os.path.abspath(calc.directory)
            msg = (
                'Calculator "{}" failed with command "{}" failed in '
                '{} with error code {}'.format(
                    calc.name, command, path, errorcode
                )
            )
            raise CalculationFailed(msg)


@dataclass
class FileIORules:
    """Rules for controlling streams options to external command.

    FileIOCalculator will direct stdin and stdout and append arguments
    to the calculator command using the specifications on this class.

    Currently names can contain "{prefix}" which will be substituted by
    calc.prefix.  This will go away if/when we can remove prefix."""
    extend_argv: Sequence[str] = tuple()
    stdin_name: Optional[str] = None
    stdout_name: Optional[str] = None

    configspec: Dict[str, Any] = field(default_factory=dict)

    def load_config(self, section):
        dct = {}
        for key, value in self.configspec.items():
            if key in section:
                value = section[key]
            dct[key] = value
        return dct


class BadConfiguration(Exception):
    pass


def _validate_command(command: str) -> str:
    # We like to store commands as strings (and call shlex.split() later),
    # but we also like to validate them early.  This will error out if
    # command contains syntax problems and will also normalize e.g.
    # multiple spaces:
    try:
        return shlex.join(shlex.split(command))
    except ValueError as err:
        raise BadConfiguration('Cannot parse command string') from err


@dataclass
class StandardProfile:
    command: str
    configvars: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.command = _validate_command(self.command)

    def execute(self, calc):
        try:
            self._call(calc, subprocess.check_call)
        except subprocess.CalledProcessError as err:
            directory = Path(calc.directory).resolve()
            msg = (f'Calculator {calc.name} failed with args {err.args} '
                   f'in directory {directory}')
            raise CalculationFailed(msg) from err

    def execute_nonblocking(self, calc):
        return self._call(calc, subprocess.Popen)

    @property
    def _split_command(self):
        # XXX Unduplicate common stuff between StandardProfile and
        # that of GenericFileIO
        return shlex.split(self.command)

    def _call(self, calc, subprocess_function):
        from contextlib import ExitStack

        directory = Path(calc.directory).resolve()
        fileio_rules = calc.fileio_rules

        with ExitStack() as stack:

            def _maybe_open(name, mode):
                if name is None:
                    return None

                name = name.format(prefix=calc.prefix)
                directory = Path(calc.directory)
                return stack.enter_context(open(directory / name, mode))

            stdout_fd = _maybe_open(fileio_rules.stdout_name, 'wb')
            stdin_fd = _maybe_open(fileio_rules.stdin_name, 'rb')

            argv = [*self._split_command, *fileio_rules.extend_argv]
            argv = [arg.format(prefix=calc.prefix) for arg in argv]
            return subprocess_function(
                argv, cwd=directory,
                stdout=stdout_fd,
                stdin=stdin_fd)


[docs]
class FileIOCalculator(Calculator):
    """Base class for calculators that write/read input/output files."""

    # Static specification of rules for this calculator:
    fileio_rules: Optional[FileIORules] = None

    # command: Optional[str] = None
    # 'Command used to start calculation'

    # Fallback command when nothing else is specified.
    # There will be no fallback in the future; it must be explicitly
    # configured.
    _legacy_default_command: Optional[str] = None

    cfg = _cfg  # Ensure easy access to config for subclasses

[docs]
    @classmethod
    def ruleset(cls, *args, **kwargs):
        """Helper for subclasses to define FileIORules."""
        return FileIORules(*args, **kwargs)


    def __init__(
        self,
        restart=None,
        ignore_bad_restart_file=Calculator._deprecated,
        label=None,
        atoms=None,
        command=None,
        profile=None,
        **kwargs,
    ):
        """File-IO calculator.

        command: str
            Command used to start calculation.
        """

        super().__init__(restart, ignore_bad_restart_file, label, atoms,
                         **kwargs)

        if profile is None:
            profile = self._initialize_profile(command)
        self.profile = profile

    @property
    def command(self):
        # XXX deprecate me
        #
        # This is for calculators that invoke Popen directly on
        # self.command instead of letting us (superclass) do it.
        return self.profile.command

    @command.setter
    def command(self, command):
        self.profile.command = command

    @classmethod
    def load_argv_profile(cls, cfg, section_name):
        # Helper method to load configuration.
        # This is used by the tests, do not rely on this as it will change.
        try:
            section = cfg.parser[section_name]
        except KeyError:
            raise BadConfiguration(f'No {section_name!r} section')

        if cls.fileio_rules is not None:
            configvars = cls.fileio_rules.load_config(section)
        else:
            configvars = {}

        try:
            command = section['command']
        except KeyError:
            raise BadConfiguration(
                f'No command field in {section_name!r} section')

        return StandardProfile(command, configvars)

    def _initialize_profile(self, command):
        if command is None:
            name = 'ASE_' + self.name.upper() + '_COMMAND'
            command = self.cfg.get(name)

        if command is None and self.name in self.cfg.parser:
            return self.load_argv_profile(self.cfg, self.name)

        if command is None:
            # XXX issue a FutureWarning if this causes the command
            # to no longer be None
            command = self._legacy_default_command

        if command is None:
            raise EnvironmentError(
                f'No configuration of {self.name}.  '
                f'Missing section [{self.name}] in configuration')

        return OldShellProfile(command)

[docs]
    def calculate(
        self, atoms=None, properties=['energy'], system_changes=all_changes
    ):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.write_input(self.atoms, properties, system_changes)
        self.execute()
        self.read_results()


    def execute(self):
        self.profile.execute(self)

[docs]
    def write_input(self, atoms, properties=None, system_changes=None):
        """Write input file(s).

        Call this method first in subclasses so that directories are
        created automatically."""

        absdir = os.path.abspath(self.directory)
        if absdir != os.curdir and not os.path.isdir(self.directory):
            os.makedirs(self.directory)

    def read_results(self):
        """Read energy, forces, ... from output file(s)."""
