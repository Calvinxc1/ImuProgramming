import serial
ser_data = []
rcd_time = td(seconds=60 * 10)
with serial.Serial(port='/dev/ttyUSB0', baudrate=115200) as ser:
    ser.readline()
    t = tqdm()
    start_time = dt.now()
    while (dt.now() - start_time) < rcd_time:
        ser_data.append(ser.readline())
        t.update()
ser_data = pd.DataFrame([data.decode().strip().split(',') for data in ser_data], columns=['milis','x','y','z']).set_index('milis').astype(float)
ser_data.index = ser_data.index.astype(int)

fig, ax = plt.subplots(3, figsize=(16,10))
for i in range(ser_data.columns.size):
    col = ser_data.columns[i]
    mean = ser_data[col].mean()
    stdev = ser_data[col].std()
    sns.distplot(ser_data[col], ax=ax[i], label='Magnetometer: {}'.format(col))
    for j in [-2, -1, 0, 1, 2]:
        ax[i].axvline(mean - (j * stdev), color='#000000')