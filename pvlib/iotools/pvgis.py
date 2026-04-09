"""
Get, read, and parse data from `PVGIS <https://ec.europa.eu/jrc/en/pvgis>`_.

For more information, see the following links:
* `Interactive Tools <https://re.jrc.ec.europa.eu/pvg_tools/en/tools.html>`_
* `Data downloads <https://ec.europa.eu/jrc/en/PVGIS/downloads/data>`_
* `User manual docs <https://ec.europa.eu/jrc/en/PVGIS/docs/usermanual>`_

More detailed information about the API for TMY and hourly radiation are here:
* `TMY <https://ec.europa.eu/jrc/en/PVGIS/tools/tmy>`_
* `hourly radiation
  <https://ec.europa.eu/jrc/en/PVGIS/tools/hourly-radiation>`_
* `daily radiation <https://ec.europa.eu/jrc/en/PVGIS/tools/daily-radiation>`_
* `monthly radiation
  <https://ec.europa.eu/jrc/en/PVGIS/tools/monthly-radiation>`_
"""
import io
import json
from dataclasses import dataclass
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import pytz
from pvlib.iotools import read_epw

URL = 'https://re.jrc.ec.europa.eu/api/'
HOURS_PER_TMY_YEAR = 8760

# Dictionary mapping PVGIS names to pvlib names
VARIABLE_MAP = {
    'G(h)': 'ghi',
    'Gb(n)': 'dni',
    'Gd(h)': 'dhi',
    'G(i)': 'poa_global',
    'Gb(i)': 'poa_direct',
    'Gd(i)': 'poa_sky_diffuse',
    'Gr(i)': 'poa_ground_diffuse',
    'H_sun': 'solar_elevation',
    'T2m': 'temp_air',
    'RH': 'relative_humidity',
    'SP': 'pressure',
    'WS10m': 'wind_speed',
    'WD10m': 'wind_direction',
}


@dataclass(frozen=True)
class _HourlyRequestOptions:
    outputformat: str
    components: bool
    surface_tilt: float
    surface_azimuth: float
    usehorizon: bool
    userhorizon: list | None
    pvcalculation: bool
    peakpower: float | None
    pvtechchoice: str
    mountingplace: str
    loss: float
    trackingtype: int
    optimal_surface_tilt: bool
    optimalangles: bool
    raddatabase: str | None
    start: object
    end: object


def _raise_for_pvgis_response(res):
    # Refactoring (Extract Method): centralized repeated PVGIS HTTP error
    # extraction/raise logic shared by multiple API functions.
    if res.ok:
        return
    try:
        err_msg = res.json()
    except Exception:
        res.raise_for_status()
    else:
        raise requests.HTTPError(err_msg['message'])


def _build_pvgis_hourly_params(latitude, longitude, options):
    # Refactoring (Introduce Parameter Object): build hourly request params
    # from a single options object instead of many independent values.
    params = {
        'lat': latitude, 'lon': longitude, 'outputformat': options.outputformat,
        'angle': options.surface_tilt, 'aspect': options.surface_azimuth - 180,
        'pvcalculation': int(options.pvcalculation),
        'pvtechchoice': options.pvtechchoice,
        'mountingplace': options.mountingplace,
        'trackingtype': options.trackingtype,
        'components': int(options.components),
        'usehorizon': int(options.usehorizon),
        'optimalangles': int(options.optimalangles),
        'optimalinclination': int(options.optimal_surface_tilt),
        'loss': options.loss,
    }
    if options.userhorizon is not None:
        params['userhorizon'] = ','.join(str(x) for x in options.userhorizon)
    if options.raddatabase is not None:
        params['raddatabase'] = options.raddatabase
    if options.start is not None:
        params['startyear'] = options.start if isinstance(options.start, int) \
            else pd.to_datetime(options.start).year
    if options.end is not None:
        params['endyear'] = options.end if isinstance(options.end, int) else \
            pd.to_datetime(options.end).year
    if options.peakpower is not None:
        params['peakpower'] = options.peakpower
    return params


def _read_pvgis_hourly_json(filename, map_variables):
    try:
        src = json.load(filename)
    except AttributeError:  # str/path has no .read() attribute
        with open(str(filename), 'r') as fbuf:
            src = json.load(fbuf)
    return _parse_pvgis_hourly_json(src, map_variables=map_variables)


def _read_pvgis_hourly_csv(filename, map_variables):
    try:
        return _parse_pvgis_hourly_csv(filename, map_variables=map_variables)
    except AttributeError:  # str/path has no .read() attribute
        with open(str(filename), 'r') as fbuf:
            return _parse_pvgis_hourly_csv(fbuf, map_variables=map_variables)


def _read_pvgis_tmy_json(filename):
    try:
        src = json.load(filename)
    except AttributeError:  # str/path has no .read() attribute
        with open(str(filename), 'r') as fbuf:
            src = json.load(fbuf)
    return _parse_pvgis_tmy_json(src)


def _read_pvgis_tmy_csv(filename):
    try:
        return _parse_pvgis_tmy_csv(filename)
    except AttributeError:  # str/path has no .read() attribute
        with open(str(filename), 'rb') as fbuf:
            return _parse_pvgis_tmy_csv(fbuf)


def _split_metadata_line(line):
    # Refactoring (Extract Method): shared metadata key/value parsing helper
    # for PVGIS CSV parser loops.
    if isinstance(line, bytes):
        line = line.decode('utf-8')
    line = line.strip()
    if ':' not in line:
        return None, None
    key, value = line.split(':', 1)
    return key, value.strip()


def get_pvgis_hourly(latitude, longitude, start=None, end=None,
                     raddatabase=None, components=True,
                     surface_tilt=0, surface_azimuth=180,
                     outputformat='json',
                     usehorizon=True, userhorizon=None,
                     pvcalculation=False,
                     peakpower=None, pvtechchoice='crystSi',
                     mountingplace='free', loss=0, trackingtype=0,
                     optimal_surface_tilt=False, optimalangles=False,
                     url=URL, map_variables=True, timeout=30):
    """Get hourly solar irradiation and modeled PV power output from PVGIS.

    PVGIS data is freely available at [1]_.

        .. versionchanged:: 0.13.0
           The function now returns two items ``(data,meta)``. Previous
           versions of this function returned three elements
           ``(data,inputs,meta)``. The ``inputs`` dictionary is now included in
           ``meta``, which has changed structure to accommodate it.

    Parameters
    ----------
    latitude: float
        In decimal degrees, between -90 and 90, north is positive (ISO 19115)
    longitude: float
        In decimal degrees, between -180 and 180, east is positive (ISO 19115)
    start : int or datetime like, optional
        First year of the radiation time series. Defaults to first year
        available.
    end : int or datetime like, optional
        Last year of the radiation time series. Defaults to last year
        available.
    raddatabase : str, optional
        Name of radiation database. Options depend on location, see [3]_.
    components: bool, default: True
        Output solar radiation components (beam, diffuse, and reflected).
        Otherwise only global irradiance is returned.
    surface_tilt: float, default: 0
        Tilt angle from horizontal plane. Ignored for two-axis tracking.
    surface_azimuth: float, default: 180
        Orientation (azimuth angle) of the (fixed) plane. Clockwise from north
        (north=0, east=90, south=180, west=270). This is offset 180 degrees from the
        convention used by PVGIS. Ignored for tracking systems.

        .. versionchanged:: 0.10.0
           The `surface_azimuth` parameter now follows the pvlib convention, which
           is clockwise from north. However, the convention used by the PVGIS website
           and pvlib<=0.9.5 is offset by 180 degrees.
    usehorizon: bool, default: True
        Include effects of horizon
    userhorizon : list of float, optional
        Optional user specified elevation of horizon in degrees, at equally
        spaced azimuth clockwise from north, only valid if ``usehorizon`` is
        true, if ``usehorizon`` is true but ``userhorizon`` is not specified then
        PVGIS will calculate the horizon [4]_
    pvcalculation: bool, default: False
        Return estimate of hourly PV production.
    peakpower : float, optional
        Nominal power of PV system in kW. Required if pvcalculation=True.
    pvtechchoice: {'crystSi', 'CIS', 'CdTe', 'Unknown'}, default: 'crystSi'
        PV technology.
    mountingplace: {'free', 'building'}, default: free
        Type of mounting for PV system. Options of 'free' for free-standing
        and 'building' for building-integrated.
    loss: float, default: 0
        Sum of PV system losses in percent. Required if pvcalculation=True
    trackingtype: {0, 1, 2, 3, 4, 5}, default: 0
        Type of suntracking. 0=fixed, 1=single horizontal axis aligned
        north-south, 2=two-axis tracking, 3=vertical axis tracking, 4=single
        horizontal axis aligned east-west, 5=single inclined axis aligned
        north-south.
    optimal_surface_tilt: bool, default: False
        Calculate the optimum tilt angle. Ignored for two-axis tracking
    optimalangles: bool, default: False
        Calculate the optimum tilt and azimuth angles. Ignored for two-axis
        tracking.
    outputformat: str, default: 'json'
        Must be in ``['json', 'csv']``. See PVGIS hourly data
        documentation [2]_ for more info.
    url: str, default: :const:`pvlib.iotools.pvgis.URL`
        Base url of PVGIS API. ``seriescalc`` is appended to get hourly data
        endpoint. Note, a specific PVGIS version can be specified, e.g.,
        https://re.jrc.ec.europa.eu/api/v5_2/
    map_variables: bool, default: True
        When true, renames columns of the Dataframe to pvlib variable names
        where applicable. See variable :const:`VARIABLE_MAP`.
    timeout: int, default: 30
        Time in seconds to wait for server response before timeout

    Returns
    -------
    data : pandas.DataFrame
        Time-series of hourly data, see Notes for fields
    metadata : dict
        Dictionary containing metadata

    Raises
    ------
    requests.HTTPError
        If the request response status is ``HTTP/1.1 400 BAD REQUEST``, then
        the error message in the response will be raised as an exception,
        otherwise raise whatever ``HTTP/1.1`` error occurred

    Hint
    ----
    PVGIS provides access to a number of different solar radiation datasets,
    including satellite-based (SARAH, SARAH2, and NSRDB PSM3) and re-analysis
    products (ERA5). Each data source has a different geographical coverage and
    time stamp convention, e.g., SARAH and SARAH2 provide instantaneous values,
    whereas values from ERA5 are averages for the hour.

    Warning
    -------
    The azimuth orientation specified in the output metadata does not
    correspond to the pvlib convention, but is offset 180 degrees. This is
    despite the fact that the input parameter `surface_tilt` has to be
    specified according to the pvlib convention.

    Notes
    -----
    data includes the following fields:

    ===========================  ======  ======================================
    raw, mapped                  Format  Description
    ===========================  ======  ======================================
    *Mapped field names are returned when the map_variables argument is True*
    ---------------------------------------------------------------------------
    P†                           float   PV system power (W)
    G(i), poa_global‡            float   Global irradiance on inclined plane (W/m^2)
    Gb(i), poa_direct‡           float   Beam (direct) irradiance on inclined plane (W/m^2)
    Gd(i), poa_sky_diffuse‡      float   Diffuse irradiance on inclined plane (W/m^2)
    Gr(i), poa_ground_diffuse‡   float   Reflected irradiance on inclined plane (W/m^2)
    H_sun, solar_elevation       float   Sun height/elevation (degrees)
    T2m, temp_air                float   Air temperature at 2 m (degrees Celsius)
    WS10m, wind_speed            float   Wind speed at 10 m (m/s)
    Int                          int     Solar radiation reconstructed (1/0)
    ===========================  ======  ======================================

    †P (PV system power) is only returned when pvcalculation=True.

    ‡Gb(i), Gd(i), and Gr(i) are returned when components=True, otherwise the
    sum of the three components, G(i), is returned.

    See Also
    --------
    pvlib.iotools.read_pvgis_hourly, pvlib.iotools.get_pvgis_tmy

    Examples
    --------
    >>> # Retrieve two years of irradiance data from PVGIS:
    >>> data, meta = pvlib.iotools.get_pvgis_hourly(  # doctest: +SKIP
    >>>    latitude=45, longitude=8, start=2015, end=2016)  # doctest: +SKIP

    References
    ----------
    .. [1] `PVGIS <https://ec.europa.eu/jrc/en/pvgis>`_
    .. [2] `PVGIS Hourly Radiation
       <https://ec.europa.eu/jrc/en/PVGIS/tools/hourly-radiation>`_
    .. [3] `PVGIS Non-interactive service
       <https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/getting-started-pvgis/api-non-interactive-service_en>`_
    .. [4] `PVGIS horizon profile tool
       <https://ec.europa.eu/jrc/en/PVGIS/tools/horizon>`_
    """  # noqa: E501
    options = _HourlyRequestOptions(
        outputformat=outputformat,
        components=components,
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        usehorizon=usehorizon,
        userhorizon=userhorizon,
        pvcalculation=pvcalculation,
        peakpower=peakpower,
        pvtechchoice=pvtechchoice,
        mountingplace=mountingplace,
        loss=loss,
        trackingtype=trackingtype,
        optimal_surface_tilt=optimal_surface_tilt,
        optimalangles=optimalangles,
        raddatabase=raddatabase,
        start=start,
        end=end,
    )
    params = _build_pvgis_hourly_params(latitude, longitude, options)

    # The url endpoint for hourly radiation is 'seriescalc'
    res = requests.get(url + 'seriescalc', params=params, timeout=timeout)
    _raise_for_pvgis_response(res)

    return read_pvgis_hourly(io.StringIO(res.text), pvgis_format=outputformat,
                             map_variables=map_variables)


def _parse_pvgis_hourly_json(src, map_variables):
    metadata = src['meta'].copy()
    # Override the "inputs" in metadata
    metadata['inputs'] = src['inputs']
    # Re-add the inputs in metadata one-layer down
    metadata['inputs']['descriptions'] = src['meta']['inputs']
    data = pd.DataFrame(src['outputs']['hourly'])
    data.index = pd.to_datetime(data['time'], format='%Y%m%d:%H%M', utc=True)
    data = data.drop('time', axis=1)
    data = data.astype(dtype={'Int': 'int'})  # The 'Int' column to be integer
    if map_variables:
        data = data.rename(columns=VARIABLE_MAP)
    return data, metadata


def _parse_pvgis_hourly_csv(src, map_variables):
    # The first 4 rows are latitude, longitude, elevation, radiation database
    metadata = {'inputs': {}, 'descriptions': {}}
    # 'location' metadata
    # 'Latitude (decimal degrees): 45.000\r\n'
    metadata['inputs']['latitude'] = float(src.readline().split(':')[1])
    # 'Longitude (decimal degrees): 8.000\r\n'
    metadata['inputs']['longitude'] = float(src.readline().split(':')[1])
    # Elevation (m): 1389.0\r\n
    metadata['inputs']['elevation'] = float(src.readline().split(':')[1])
    # 'Radiation database: \tPVGIS-SARAH\r\n'
    metadata['inputs']['radiation_database'] = \
        src.readline().split(':')[1].strip()
    # Parse through the remaining metadata section (the number of lines for
    # this section depends on the requested parameters)
    while True:
        line = src.readline()
        if line.startswith('time,'):  # The data header starts with 'time,'
            # The last line of the metadata section contains the column names
            names = line.strip().split(',')
            break
        # Only retrieve metadata from non-empty lines
        elif line.strip() != '':
            key, value = _split_metadata_line(line)
            if key is not None:
                metadata['inputs'][key] = value
        elif line == '':  # If end of file is reached
            raise ValueError('No data section was detected. File has probably '
                             'been modified since being downloaded from PVGIS')
    # Save the entries from the data section to a list, until an empty line is
    # reached an empty line. The length of the section depends on the request
    data_lines = []
    while True:
        line = src.readline()
        if line.strip() == '':
            break
        else:
            data_lines.append(line.strip().split(','))
    data = pd.DataFrame(data_lines, columns=names)
    data.index = pd.to_datetime(data['time'], format='%Y%m%d:%H%M', utc=True)
    data = data.drop('time', axis=1)
    if map_variables:
        data = data.rename(columns=VARIABLE_MAP)
    # All columns should have the dtype=float, except 'Int' which should be
    # integer. It is necessary to convert to float, before converting to int
    data = data.astype(float).astype(dtype={'Int': 'int'})
    # Generate metadata dictionary containing description of parameters
    metadata['descriptions'] = {}
    for line in src.readlines():
        key, value = _split_metadata_line(line)
        if key is not None:
            metadata['descriptions'][key] = value
    return data, metadata


def read_pvgis_hourly(filename, pvgis_format=None, map_variables=True):
    """Read a PVGIS hourly file.

        .. versionchanged:: 0.13.0
           The function now returns two items ``(data,meta)``. Previous
           versions of this function returned three elements
           ``(data,inputs,meta)``. The ``inputs`` dictionary is now included in
           ``meta``, which has changed structure to accommodate it.

    Parameters
    ----------
    filename : str, pathlib.Path, or file-like buffer
        Name, path, or buffer of hourly data file downloaded from PVGIS.
    pvgis_format : str, optional
        Format of PVGIS file or buffer. Equivalent to the ``outputformat``
        parameter in the PVGIS API. If ``filename`` is a file and
        ``pvgis_format`` is not specified then the file extension will be used
        to determine the PVGIS format to parse. If ``filename`` is a buffer,
        then ``pvgis_format`` is required and must be in ``['csv', 'json']``.
    map_variables: bool, default True
        When true, renames columns of the DataFrame to pvlib variable names
        where applicable. See variable :const:`VARIABLE_MAP`.

    Returns
    -------
    data : pandas.DataFrame
        the time series data
    metadata : dict
        metadata

    Warning
    -------
    The azimuth orientation specified in the output metadata does not
    correspond to the pvlib convention, but is offset 180 degrees.

    Raises
    ------
    ValueError
        if ``pvgis_format`` is not specified and the file extension is neither
        ``.csv`` nor ``.json`` or if ``pvgis_format`` is provided as
        input but isn't in ``['csv', 'json']``
    TypeError
        if ``pvgis_format`` is not specified and ``filename`` is a buffer

    See Also
    --------
    get_pvgis_hourly, read_pvgis_tmy
    """
    # get the PVGIS outputformat
    if pvgis_format is None:
        # get the file extension from suffix, but remove the dot and make sure
        # it's lower case to compare with csv, or json
        # NOTE: basic format is not supported for PVGIS Hourly as the data
        # format does not include a header
        # NOTE: raises TypeError if filename is a buffer
        outputformat = Path(filename).suffix[1:].lower()
    else:
        outputformat = pvgis_format

    # parse the pvgis file based on the output format, either 'json' or 'csv'
    # NOTE: json and csv output formats have parsers defined as private
    # functions in this module

    # Refactoring (Replace Conditional with Dispatch Table): parser selection
    # now uses format->reader mapping instead of if/elif branches.
    parser_dispatch = {
        'json': _read_pvgis_hourly_json,
        'csv': _read_pvgis_hourly_csv,
    }
    parser = parser_dispatch.get(outputformat)
    if parser is None:
        # raise exception if pvgis format isn't in ['csv', 'json']
        err_msg = (
            "pvgis format '{:s}' was unknown, must be either 'json' or 'csv'")\
            .format(outputformat)
        raise ValueError(err_msg)
    return parser(filename, map_variables)


def _coerce_and_roll_tmy(tmy_data, tz, year):
    """
    Assumes ``tmy_data`` input is UTC, converts from UTC to ``tz``, rolls
    dataframe so timeseries starts at midnight, and forces all indices to
    ``year``. Only works for integer ``tz``, but ``None`` and ``False`` are
    re-interpreted as zero / UTC.
    """
    if tz:
        tzname = pytz.timezone(f'Etc/GMT{-tz:+d}')
    else:
        tz = 0
        tzname = pytz.timezone('UTC')
    new_index = pd.DatetimeIndex([
        timestamp.replace(year=year, tzinfo=tzname)
        for timestamp in tmy_data.index],
        name=f'time({tzname})')
    new_tmy_data = pd.DataFrame(
        np.roll(tmy_data, tz, axis=0),
        columns=tmy_data.columns,
        index=new_index)
    # GH 2399
    new_tmy_data = \
        new_tmy_data.astype(dtype=dict(zip(tmy_data.columns, tmy_data.dtypes)))
    return new_tmy_data


def get_pvgis_tmy(latitude, longitude, outputformat='json', usehorizon=True,
                  userhorizon=None, startyear=None, endyear=None,
                  map_variables=True, url=URL, timeout=30,
                  roll_utc_offset=None, coerce_year=1990):
    """
    Get TMY data from PVGIS.

    For more information see the PVGIS [1]_ TMY tool documentation [2]_.

        .. versionchanged:: 0.13.0
           The function now returns two items ``(data,meta)``. Previous
           versions of this function returned four elements
           ``(data,months_selected,inputs,meta)``. The ``inputs`` dictionary
           and ``months_selected`` are  now included in ``meta``, which has
           changed structure to accommodate it.

    Parameters
    ----------
    latitude : float
        Latitude in degrees north
    longitude : float
        Longitude in degrees east
    outputformat : str, default 'json'
        Must be in ``['csv', 'epw', 'json']``. See PVGIS TMY tool
        documentation [2]_ for more info.
    usehorizon : bool, default True
        include effects of horizon
    userhorizon : list of float, optional
        Optional user-specified elevation of horizon in degrees, at equally
        spaced azimuth clockwise from north. If not specified, PVGIS will
        calculate the horizon [3]_. If specified, requires ``usehorizon=True``.
    startyear : int, optional
        first year to calculate TMY
    endyear : int, optional
        last year to calculate TMY, must be at least 10 years from first year
    map_variables: bool, default True
        When true, renames columns of the Dataframe to pvlib variable names
        where applicable. See variable :const:`VARIABLE_MAP`.
    url : str, default: :const:`pvlib.iotools.pvgis.URL`
        base url of PVGIS API, append ``tmy`` to get TMY endpoint
    timeout : int, default 30
        time in seconds to wait for server response before timeout
    roll_utc_offset: int, optional
        Use to specify a time zone other than the default UTC zero and roll
        dataframe by ``roll_utc_offset`` so it starts at midnight on January
        1st. Ignored if ``None``, otherwise will force year to ``coerce_year``.
    coerce_year: int, default 1990
        Use to force indices to desired year. Defaults to 1990. Specify
        ``None`` to return the actual indices used for the TMY. If
        ``coerce_year`` is ``None``, but ``roll_utc_offset`` is specified,
        then ``coerce_year`` will be set to the default.

    Returns
    -------
    data : pandas.DataFrame
        the weather data
    metadata : list or dict
        file metadata

    Raises
    ------
    requests.HTTPError
        if the request response status is ``HTTP/1.1 400 BAD REQUEST``, then
        the error message in the response will be raised as an exception,
        otherwise raise whatever ``HTTP/1.1`` error occurred

    See Also
    --------
    read_pvgis_tmy

    References
    ----------
    .. [1] `PVGIS <https://ec.europa.eu/jrc/en/pvgis>`_
    .. [2] `PVGIS TMY tool <https://ec.europa.eu/jrc/en/PVGIS/tools/tmy>`_
    .. [3] `PVGIS horizon profile tool
       <https://ec.europa.eu/jrc/en/PVGIS/tools/horizon>`_
    """
    # use requests to format the query string by passing params dictionary
    params = {'lat': latitude, 'lon': longitude, 'outputformat': outputformat}
    # pvgis only likes 0 for False, and 1 for True, not strings, also the
    # default for usehorizon is already 1 (ie: True), so only set if False
    if not usehorizon:
        params['usehorizon'] = 0
    if userhorizon is not None:
        params['userhorizon'] = ','.join(str(x) for x in userhorizon)
    if startyear is not None:
        params['startyear'] = startyear
    if endyear is not None:
        params['endyear'] = endyear
    res = requests.get(url + 'tmy', params=params, timeout=timeout)
    _raise_for_pvgis_response(res)
    # initialize data to None in case API fails to respond to bad outputformat
    data = None, None
    if outputformat == 'json':
        src = res.json()
        data, meta = _parse_pvgis_tmy_json(src)
    elif outputformat == 'csv':
        with io.BytesIO(res.content) as src:
            data, meta = _parse_pvgis_tmy_csv(src)
    elif outputformat == 'epw':
        with io.StringIO(res.content.decode('utf-8')) as src:
            data, meta = read_epw(src)
    elif outputformat == 'basic':
        err_msg = ("outputformat='basic' is no longer supported by pvlib, "
                   "please use outputformat='csv' instead.")
        raise ValueError(err_msg)

    if map_variables:
        data = data.rename(columns=VARIABLE_MAP)

    if not (roll_utc_offset is None and coerce_year is None):
        # roll_utc_offset is specified, but coerce_year isn't
        coerce_year = coerce_year or 1990
        data = _coerce_and_roll_tmy(data, roll_utc_offset, coerce_year)

    return data, meta


def _parse_pvgis_tmy_json(src):
    meta = src['meta'].copy()
    # Override the "inputs" in metadata
    meta['inputs'] = src['inputs']
    # Re-add the inputs in metadata one-layer down
    meta['inputs']['descriptions'] = src['meta']['inputs']
    meta['months_selected'] = src['outputs']['months_selected']
    data = pd.DataFrame(src['outputs']['tmy_hourly'])
    data.index = pd.to_datetime(
        data['time(UTC)'], format='%Y%m%d:%H%M', utc=True)
    data = data.drop('time(UTC)', axis=1)
    return data, meta


def _parse_pvgis_tmy_csv(src):
    # the first 3 rows are latitude, longitude, elevation
    meta = {'inputs': {}, 'descriptions': {}}
    # 'Latitude (decimal degrees): 45.000\r\n'
    meta['inputs']['latitude'] = float(src.readline().split(b':')[1])
    # 'Longitude (decimal degrees): 8.000\r\n'
    meta['inputs']['longitude'] = float(src.readline().split(b':')[1])
    # Elevation (m): 1389.0\r\n
    meta['inputs']['elevation'] = float(src.readline().split(b':')[1])

    # TMY has an extra line here: Irradiance Time Offset (h): 0.1761\r\n
    line = src.readline()
    if line.startswith(b'Irradiance Time Offset'):
        meta['inputs']['irradiance time offset'] = float(line.split(b':')[1])
        src.readline()  # skip over the "month,year\r\n"
    else:
        # `line` is already the "month,year\r\n" line, so nothing to do
        pass
    # then there's a 13 row comma separated table with two columns: month, year
    # which contains the year used for that month in the TMY
    months_selected = []
    for month in range(12):
        months_selected.append(
            {'month': month+1, 'year': int(src.readline().split(b',')[1])})
    meta['months_selected'] = months_selected
    # then there's the TMY (typical meteorological year) data
    # first there's a header row:
    #    time(UTC),T2m,RH,G(h),Gb(n),Gd(h),IR(h),WS10m,WD10m,SP
    headers = [h.decode('utf-8').strip() for h in src.readline().split(b',')]
    # Refactoring (Replace Magic Number with Named Constant): use explicit TMY
    # row-count constant instead of hard-coded 8760 for PVGIS hourly TMY data.
    data = pd.DataFrame(
        [src.readline().split(b',') for _ in range(HOURS_PER_TMY_YEAR)],
        columns=headers)
    dtidx = data['time(UTC)'].apply(lambda dt: dt.decode('utf-8'))
    dtidx = pd.to_datetime(dtidx, format='%Y%m%d:%H%M', utc=True)
    data = data.drop('time(UTC)', axis=1)
    data = pd.DataFrame(data, dtype=float)
    data.index = dtidx
    # finally there's some meta data
    meta['descriptions'] = {}
    for line in src.readlines():
        key, value = _split_metadata_line(line)
        if key is not None:
            meta['descriptions'][key] = value
    return data, meta


def read_pvgis_tmy(filename, pvgis_format=None, map_variables=True):
    """
    Read a TMY file downloaded from PVGIS.

        .. versionchanged:: 0.13.0
           The function now returns two items ``(data,meta)``. Previous
           versions of this function returned four elements
           ``(data,months_selected,inputs,meta)``. The ``inputs`` dictionary
           and ``months_selected`` are  now included in ``meta``, which has
           changed structure to accommodate it.

    Parameters
    ----------
    filename : str, pathlib.Path, or file-like buffer
        Name, path, or buffer of file downloaded from PVGIS.
    pvgis_format : str, optional
        Format of PVGIS file or buffer. Equivalent to the ``outputformat``
        parameter in the PVGIS TMY API. If ``filename`` is a file and
        ``pvgis_format`` is not specified then the file extension will be used
        to determine the PVGIS format to parse.
        If ``filename`` is a buffer, then ``pvgis_format`` is required and must
        be in ``['csv', 'epw', 'json']``.
    map_variables: bool, default True
        When true, renames columns of the Dataframe to pvlib variable names
        where applicable. See variable :const:`VARIABLE_MAP`.


    Returns
    -------
    data : pandas.DataFrame
        the weather data
    metadata : list or dict
        file metadata

    Raises
    ------
    ValueError
        if ``pvgis_format`` is not specified and the file extension is neither
        ``.csv``, ``.json``, nor ``.epw``, or if ``pvgis_format`` is provided
        as input but isn't in ``['csv', 'epw', 'json']``
    TypeError
        if ``pvgis_format`` is not specified and ``filename`` is a buffer

    See Also
    --------
    get_pvgis_tmy
    """
    # get the PVGIS outputformat
    if pvgis_format is None:
        # get the file extension from suffix, but remove the dot and make sure
        # it's lower case to compare with epw, csv, or json
        # NOTE: raises TypeError if filename is a buffer
        outputformat = Path(filename).suffix[1:].lower()
    else:
        outputformat = pvgis_format
    # parse pvgis file based on outputformat, either 'epw', 'json', or 'csv'

    if outputformat == 'basic':
        err_msg = "outputformat='basic' is no longer supported, please use " \
            "outputformat='csv' instead."
        raise ValueError(err_msg)
    # Refactoring (Replace Conditional with Dispatch Table): parser selection
    # now uses format->reader mapping instead of if/elif branches.
    parser_dispatch = {
        'epw': read_epw,
        'json': _read_pvgis_tmy_json,
        'csv': _read_pvgis_tmy_csv,
    }
    parser = parser_dispatch.get(outputformat)
    if parser is None:
        # raise exception if pvgis format isn't in ['csv','epw','json']
        err_msg = (
            "pvgis format '{:s}' was unknown, must be either 'json', 'csv',"
            "or 'epw'").format(outputformat)
        raise ValueError(err_msg)
    data, meta = parser(filename)

    if map_variables:
        data = data.rename(columns=VARIABLE_MAP)

    return data, meta


def get_pvgis_horizon(latitude, longitude, url=URL, **kwargs):
    """Get horizon data from PVGIS.

    Parameters
    ----------
    latitude : float
        Latitude in degrees north
    longitude : float
        Longitude in degrees east
    url: str, default: :const:`pvlib.iotools.pvgis.URL`
        Base URL for PVGIS
    kwargs:
        Passed to requests.get

    Returns
    -------
    data : pd.Series
        Pandas Series of the retrieved horizon elevation angles. Index is the
        corresponding horizon azimuth angles.
    metadata : dict
        Metadata returned by PVGIS.

    Notes
    -----
    The horizon azimuths are specified clockwise from north, e.g., south=180.
    This is the standard pvlib convention, although the PVGIS website specifies
    south=0.

    References
    ----------
    .. [1] `PVGIS horizon profile tool
       <https://ec.europa.eu/jrc/en/PVGIS/tools/horizon>`_
    """
    params = {'lat': latitude, 'lon': longitude, 'outputformat': 'json'}
    res = requests.get(url + 'printhorizon', params=params, **kwargs)
    _raise_for_pvgis_response(res)
    json_output = res.json()
    metadata = json_output['meta']
    data = pd.DataFrame(json_output['outputs']['horizon_profile'])
    data.columns = ['horizon_azimuth', 'horizon_elevation']
    # Convert azimuth to pvlib convention (north=0, south=180)
    data['horizon_azimuth'] += 180
    data.set_index('horizon_azimuth', inplace=True)
    data = data['horizon_elevation']  # convert to pd.Series
    data = data[data.index < 360]  # remove duplicate north point (0 and 360)
    return data, metadata
