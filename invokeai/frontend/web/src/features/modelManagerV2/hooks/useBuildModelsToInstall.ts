import { EMPTY_ARRAY } from 'app/store/constants';
import { useCallback } from 'react';
import { modelConfigsAdapterSelectors, useGetModelConfigsQuery } from 'services/api/endpoints/models';
import type { StarterModel } from 'services/api/types';

type ModelInstallArg = {
  config: Pick<StarterModel, 'name' | 'base' | 'type' | 'description' | 'format' | 'variant'>;
  source: string;
};

/**
 * Flattens a starter model and its dependencies into a list of models, including the starter model itself.
 */
export const flattenStarterModel = (starterModel: StarterModel): StarterModel[] => {
  return [starterModel, ...(starterModel.dependencies || [])];
};

export const useBuildModelInstallArg = () => {
  const { modelList } = useGetModelConfigsQuery(undefined, {
    selectFromResult: ({ data }) => ({ modelList: data ? modelConfigsAdapterSelectors.selectAll(data) : EMPTY_ARRAY }),
  });

  const getIsInstalled = useCallback(
    ({ source, name, base, type, is_installed, previous_names }: StarterModel): boolean =>
      modelList.some(
        (mc) =>
          is_installed ||
          source === mc.source ||
          (base === mc.base && (name === mc.name || previous_names?.includes(name)) && type === mc.type)
      ),
    [modelList]
  );

  const buildModelInstallArg = useCallback((starterModel: StarterModel): ModelInstallArg => {
    const { name, base, type, source, description, format, variant } = starterModel;

    const config: Record<string, unknown> = {
      name,
      base,
      type,
      description,
      format,
      variant,
    };

    const quantType = (starterModel as StarterModel & { quant_type?: string | null }).quant_type;
    const groupSize = (starterModel as StarterModel & { group_size?: number | null }).group_size;
    if (quantType !== undefined && quantType !== null) {
      (config as Record<string, unknown>).quant_type = quantType;
    }
    if (groupSize !== undefined && groupSize !== null) {
      (config as Record<string, unknown>).group_size = groupSize;
    }

    return {
      config: config as ModelInstallArg['config'],
      source,
    };
  }, []);

  return { getIsInstalled, buildModelInstallArg };
};
